#!/usr/bin/env python3

import argparse
import functools
import json
import re
import subprocess
import sys
import typing
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field

CYCLE_REGEX = re.compile(r"\s*([0-9]+) ([0-9]+)\s+[0-9]+\s+(0x[0-9a-fz]+)\s+DASM\(([0-9a-fz]+)\)\s*#;(.*)")


class Instruction:
    raw_encoding: int

    __slots__ = ['raw_encoding']

    def __new__(cls, raw_encoding: int) -> 'Instruction':
        if cls is not Instruction:
            instance = super().__new__(cls)
            instance.raw_encoding = raw_encoding
            return instance

        candidate = None
        opcode = raw_encoding & 0x7f
        for subclass in cls.__subclasses__():
            if opcode in subclass.OP_CODES:
                if candidate is None:
                    candidate = subclass(raw_encoding)
                else:
                    # Use the most derived instruction.
                    new_candidate = subclass(raw_encoding)
                    if type(candidate).mro().index(Instruction) \
                            < type(new_candidate).mro().index(Instruction):
                        candidate = new_candidate

        if candidate is not None:
            return candidate

        instance = super().__new__(cls)
        instance.raw_encoding = raw_encoding
        return instance

    @property
    def opcode(self) -> int:
        return self.raw_encoding & 0x7f

    @classmethod
    def read_rs1(cls, state: 'TraceState', signed=False):
        value = state.raw_cpu_state['opa']
        if signed:
            value = _bitpattern_to_signed(value, bitwidth=32)
        return value


class CSRInstruction(Instruction):
    OP_CODES = {0b1110011}

    @property
    def funct3(self) -> int:
        return (self.raw_encoding >> 12) & 0b111

    @property
    def is_csrrsi(self):
        return self.funct3 == 0b110

    @property
    def is_csrrci(self):
        return self.funct3 == 0b111

    @property
    def csr(self):
        return self.raw_encoding >> 20


def _bitpattern_to_signed(value: int, *, bitwidth: int) -> int:
    if value >= 2 ** (bitwidth - 1):
        value -= 2 ** bitwidth
    return value


class RInstruction(Instruction):
    OP_CODES = {0x2b}

    def __new__(cls, raw_encoding: int) -> 'RInstruction':
        if cls is not RInstruction and issubclass(cls, RInstruction):
            return super().__new__(cls, raw_encoding)

        r_inst = super().__new__(cls, raw_encoding)
        for subclass in cls.__subclasses__():
            if r_inst.opcode == subclass.OPCODE \
                    and r_inst.funct3 == subclass.FUNCT3 \
                    and r_inst.funct7 == subclass.FUNCT7:
                return subclass(raw_encoding)

        return r_inst

    @classmethod
    def read_rs2(cls, state: 'TraceState'):
        return state.raw_cpu_state['opb']

    @property
    def rs2_imm5(self):
        return (self.raw_encoding >> 20) & 0x1f

    @property
    def funct3(self) -> int:
        return (self.raw_encoding >> 12) & 0x7

    @property
    def funct7(self) -> int:
        return (self.raw_encoding >> 25) & 0x7f


class IInstruction(Instruction):
    OP_CODES = {0x2b}

    def __new__(cls, raw_encoding: int) -> 'IInstruction':
        if cls is not IInstruction and issubclass(cls, IInstruction):
            return super().__new__(cls, raw_encoding)

        i_inst = super().__new__(cls, raw_encoding)
        for subclass in cls.__subclasses__():
            if i_inst.opcode == subclass.OPCODE \
                    and i_inst.funct3 == subclass.FUNCT3:
                return subclass(raw_encoding)

        return i_inst

    @property
    def imm(self):
        return (self.raw_encoding >> 20) & 0xfff

    @property
    def funct3(self) -> int:
        return (self.raw_encoding >> 12) & 0x7


class DMSRCInstruction(RInstruction):
    OPCODE = 0x2b
    FUNCT3 = 0
    FUNCT7 = 0

    read_source = RInstruction.read_rs1


class DMDSTInstruction(RInstruction):
    OPCODE = 0x2b
    FUNCT3 = 0
    FUNCT7 = 0b1

    read_destination = RInstruction.read_rs1


class DMREPInstruction(RInstruction):
    OPCODE = 0x2b
    FUNCT3 = 0
    FUNCT7 = 0b111

    read_reps = RInstruction.read_rs1


class DMSTRInstruction(RInstruction):
    OPCODE = 0x2b
    FUNCT3 = 0
    FUNCT7 = 0b110

    read_source_strides = RInstruction.read_rs1
    read_dest_strides = RInstruction.read_rs2


class DMCPYIInstruction(RInstruction):
    OPCODE = 0x2b
    FUNCT3 = 0
    FUNCT7 = 0b10

    read_size = RInstruction.read_rs1
    config = RInstruction.rs2_imm5

    @property
    def is_2d(self):
        return (self.config & 0b10) != 0


class DMSTATIInstruction(RInstruction):
    OPCODE = 0x2b
    FUNCT3 = 0
    FUNCT7 = 0b100

    status = RInstruction.rs2_imm5


class SCFGWIInstruction(IInstruction):
    OPCODE = 0x2b
    FUNCT3 = 0b010

    read_value = IInstruction.read_rs1

    @property
    def ssr(self):
        return self.imm & 0x1F

    @property
    def reg(self):
        return (self.imm >> 5) & 0x3F


class TraceState:
    clock_cycle: int
    # PC or None if in FPSS sequencer.
    pc: int | None
    # None if an frep.
    instruction: Instruction | None
    raw_cpu_state: dict

    def __init__(self, clock_cycle, pc, op_code, cpu_state):
        self.clock_cycle = clock_cycle
        self.pc = pc
        self.instruction = None if op_code is None else Instruction(op_code)
        self.raw_cpu_state = cpu_state

    @property
    def in_fpss_sequencer(self):
        return self.pc is None

    @property
    def stalled(self) -> bool:
        return bool(self.raw_cpu_state.get('stall'))


def get_trace_state(line: str):
    match = CYCLE_REGEX.match(line)
    if match is None:
        return None

    pc = match.group(3)
    # No program counter if we are within the sequencer.
    if pc == '0xzzzzzzzz':
        pc = None
    else:
        pc = int(pc, base=16)

    op_code = match.group(4)
    # freps in the sequencer don't have an opcode.
    if op_code == 'zzzzzzzzzzzzzzzz':
        op_code = None
    else:
        op_code = int(op_code, base=16)

    return TraceState(int(match.group(2), base=10), pc, op_code,
                      eval(match.group(5)))


class KernelNameResolver:
    elf: str
    addr2line: str
    traces: tuple[typing.IO]

    def __init__(self, elf, addr2line, traces: tuple[str]):
        self.elf = elf
        self.addr2line = addr2line
        self.traces = (*map(lambda t: open(t, 'r'), traces),)

    def __exit__(self, exc_type, exc_val, exc_tb):
        map(lambda t: t.close(), self.traces)

    @functools.cache
    def get_name_from_address(self, address: int):
        p = subprocess.run([self.addr2line, '-e', self.elf, '-f', hex(address)], check=True, capture_output=True,
                           text=True)
        return p.stdout.splitlines()[0]

    def get_name(self, cycle: int, hartid: int):
        pattern = re.compile(r"\s*[0-9]+ " + str(cycle - 1))
        iterator = self.traces[hartid]
        for l in iterator:
            if pattern.match(l):
                break
        else:
            return "<unknown-kernel>"

        for index, l in enumerate(iterator):
            # Give up.
            if index == 100:
                return "<unknown-kernel>"

            res = get_trace_state(l)
            if res is None:
                return "<unknown-kernel>"

            if not res.stalled and res.raw_cpu_state['pc_d'] != res.pc + 4:
                return self.get_name_from_address(res.raw_cpu_state['pc_d'])


class Event(ABC):

    @abstractmethod
    def to_chrome_tracing(self, hartid: int) -> dict:
        ...


class DurationEvent(Event):
    name: str
    cycle_start: int
    cycle_duration: int
    categories: list[str]
    args: dict

    def __init__(self, name, cycle_start, cycle_duration, categories, args=None):
        if args is None:
            args = {}

        self.name = name
        self.cycle_start = cycle_start
        self.cycle_duration = cycle_duration
        self.categories = categories
        self.args = args

    def to_chrome_tracing(self, hartid: int) -> dict:
        return {
            'name': self.name,
            'ts': self.cycle_start,
            'dur': self.cycle_duration,
            'ph': 'X',
            'cat': ','.join(self.categories),
            'pid': 0,
            'tid': hartid,
            'args': self.args,
        }


class InstantEvent(Event):
    name: str
    cycle_occurred: int
    categories: list[str]
    args: dict

    def __init__(self, name, cycle_occurred, categories, args=None):
        if args is None:
            args = {}

        self.name = name
        self.cycle_occurred = cycle_occurred
        self.categories = categories
        self.args = args

    def to_chrome_tracing(self, hartid: int) -> dict:
        return {
            'name': self.name,
            'ts': self.cycle_occurred,
            'ph': 'i',
            'cat': ','.join(self.categories),
            'pid': 0,
            'tid': hartid,
            'args': self.args,
            's': 't'
        }


class KernelEvent(DurationEvent):

    def __init__(self, name, cycle_start, cycle_duration, is_dm_core, origin, metrics):
        super().__init__(name, cycle_start, cycle_duration, ['vm'] if is_dm_core else ['kernel', origin], metrics)


def calculate_sections(inputs: typing.Iterable[typing.IO], elf: str, addr2line: str,
                       traces: typing.Iterable[str]) -> \
        list[dict]:
    resolver = KernelNameResolver(elf, addr2line, (*traces,))

    # TraceViewer events
    events = []
    origin = None
    for hartid, file in enumerate(inputs):
        is_dm_core = hartid == 8

        j = json.load(file)
        for index, section in enumerate(j):
            # In the case of the DMA core we are interested in time spent between kernels to measure overhead of
            # IREE abstractions.
            if not is_dm_core:
                if index % 2 == 0:
                    continue
            else:
                if index % 2 == 1:
                    continue

            name = "vm"
            if not is_dm_core:
                name = resolver.get_name(section['start'], hartid)
                origin = "xDSL" if name.endswith('$iree_to_xdsl') else "LLVM"
                name = name.removesuffix('$iree_to_xdsl')

            start = section['start']
            dur = section['end'] - section['start']
            events.append(KernelEvent(name, start, dur, is_dm_core, origin, section).to_chrome_tracing(hartid))

    return events


class EventGenerator(ABC):

    def __init__(self):
        self._acc_callback = []

    def schedule_writeback(self, state: TraceState, callback: typing.Callable[[int], list[Event]]) -> list[Event]:
        if state.raw_cpu_state['write_rd'] and state.raw_cpu_state['rd'] != 0:
            # The result is written back in the same cycle.
            return callback(state.raw_cpu_state['writeback'])

        # Result will only be assigned at a later cycle (FIFO style).
        self._acc_callback.append(callback)
        return []

    def check_accumulator(self, state: TraceState):
        if not self._acc_callback:
            return []

        if not state.raw_cpu_state['retire_acc'] or state.raw_cpu_state['acc_pid'] == 0:
            return []

        front = self._acc_callback.pop(0)
        return front(state.raw_cpu_state['acc_pdata_32'])

    @abstractmethod
    def cycle(self, state: TraceState) -> list[dict]:
        ...


class BarrierWaitEvent(DurationEvent):
    def __init__(self, cycle_start, cycle_duration, pc):
        super().__init__('barrier', cycle_start, cycle_duration, ['barrier'], {
            'program counter': hex(pc)
        })


class BarrierUnlockEvent(InstantEvent):
    def __init__(self, cycle_occurred, pc):
        super().__init__('barrier unlocked', cycle_occurred, ['barrier'], {
            'program counter': hex(pc)
        })


class BarrierEventGenerator(EventGenerator):
    _INSTANT_THRESHOLD = 4
    _barrier_start_state: TraceState | None = None

    def cycle(self, state: TraceState) -> list[BarrierWaitEvent]:
        result = []

        if self._barrier_start_state is not None:
            # In sequencer.
            if state.pc is None:
                return result

            if state.clock_cycle - self._barrier_start_state.clock_cycle > self._INSTANT_THRESHOLD:
                result.append(BarrierWaitEvent(self._barrier_start_state.clock_cycle,
                                               state.clock_cycle - self._barrier_start_state.clock_cycle,
                                               self._barrier_start_state.pc))
            else:
                result.append(BarrierUnlockEvent(state.clock_cycle, self._barrier_start_state.pc))

            self._barrier_start_state = None

        csr_addr = state.raw_cpu_state.get('csr_addr')
        if csr_addr != 0x7c2:
            return result

        self._barrier_start_state = state
        return result


@dataclass
class _SSRState:
    bounds: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    strides: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    repetition: int = 1
    address: int = 0
    rank: int = 0
    write_stream: bool = False


class StreamingEvent(DurationEvent):
    def __init__(self, cycle_start, cycle_duration, ssr_states: tuple[_SSRState, _SSRState, _SSRState]):
        # Post process it to a human-readable format rather than the internal IR of the core that the ASM instructions
        # deal with.
        copy = deepcopy(ssr_states)
        for ssr_state in copy:
            del ssr_state.bounds[ssr_state.rank:]
            del ssr_state.strides[ssr_state.rank:]

            # The strides are in a format where they represent the offset that need to be applied when at the end of a
            # dimension, to get to the next dimension. We want it in the same form as in MLIR where it is the difference
            # between the increase of 1 in a dimension.
            size = 0
            for i in range(len(ssr_state.strides)):
                ssr_state.strides[i] += size
                size += ssr_state.strides[i] * ssr_state.bounds[i]

            for i in range(len(ssr_state.bounds)):
                ssr_state.bounds[i] += 1

        super().__init__('streaming', cycle_start, cycle_duration, ['streaming'], {
            'SSR State': [
                {
                    'bounds': json.dumps(ssr_state.bounds),
                    'strides in bytes': json.dumps(ssr_state.strides),
                    'repetition': ssr_state.repetition,
                    'address': hex(ssr_state.address),
                    'rank': ssr_state.rank,
                    'is a write stream': ssr_state.write_stream
                }
                for ssr_state in copy
            ]
        })


class StreamingEventGenerator(EventGenerator):
    _stream_start_state: TraceState | None = None
    _ssr_states: tuple[_SSRState, _SSRState, _SSRState]
    _active_ssr_states: tuple[_SSRState, _SSRState, _SSRState] | None = None

    def __init__(self):
        super().__init__()
        self._ssr_states = (_SSRState(), _SSRState(), _SSRState())

    def cycle(self, state: TraceState) -> list[StreamingEvent]:
        result = []

        if isinstance(state.instruction, SCFGWIInstruction) and not state.in_fpss_sequencer:
            ssr_states = self._ssr_states if state.instruction.ssr == 0x1F else (
                self._ssr_states[state.instruction.ssr],)
            reg = state.instruction.reg
            for ssr_state in ssr_states:
                if reg == 1:
                    ssr_state.repetition = SCFGWIInstruction.read_value(state)
                elif 2 <= reg < 6:
                    ssr_state.bounds[reg - 2] = SCFGWIInstruction.read_value(state)
                elif 6 <= reg < 10:
                    ssr_state.strides[reg - 6] = SCFGWIInstruction.read_value(state, signed=True)
                elif 24 <= reg < 28:
                    ssr_state.rank = reg - 23
                    ssr_state.write_stream = False
                    ssr_state.address = SCFGWIInstruction.read_value(state)
                elif 28 <= reg < 32:
                    ssr_state.rank = reg - 27
                    ssr_state.write_stream = True
                    ssr_state.address = SCFGWIInstruction.read_value(state)

        if not isinstance(state.instruction, CSRInstruction) or state.instruction.csr != 0x7c0:
            return result

        # Stream enables and disables always appear twice. Once when issued from the integer core, second when processed
        # by the FPSS. We denote the start as when issued by the integer core and the end when disabled in the FPSS.
        if state.instruction.is_csrrsi:
            if self._stream_start_state is None:
                self._stream_start_state = state
                self._active_ssr_states = deepcopy(self._ssr_states)
        elif state.instruction.is_csrrci and state.in_fpss_sequencer:
            if self._stream_start_state is not None and self._active_ssr_states is not None:
                result.append(StreamingEvent(self._stream_start_state.clock_cycle,
                                             state.clock_cycle - self._stream_start_state.clock_cycle,
                                             self._active_ssr_states))

            self._active_ssr_states = None
            self._stream_start_state = None

        return result


class DMAEvent(DurationEvent):
    def __init__(self, cycle_start, cycle_duration, source: int, destination: int, inner_loop: int, transfer_id: int,
                 is_2d: bool, source_strides: int, dest_strides: int, outer_loop: int):
        byte_count = inner_loop
        extra_kw = {}
        if is_2d:
            byte_count *= outer_loop
            extra_kw = {
                'source strides': source_strides,
                'destination strides': dest_strides,
                'inner loop': inner_loop,
                'outer loop': outer_loop,
            }

        super().__init__('DMA Transfer', cycle_start, cycle_duration, ['DMA'], {
            'source': hex(source),
            'destination': hex(destination),
            'copied bytes': byte_count,
            'id': transfer_id,
            '2d': is_2d,
            'Bytes per Cycle': byte_count / cycle_duration,
            **extra_kw
        })


class DMAEventGenerator(EventGenerator):
    @dataclass
    class _InFlightDMA:
        started: int
        source: int
        destination: int
        inner_loop: int
        is_2d: bool
        source_strides: int = 0
        dest_strides: int = 0
        outer_loop: int = 0
        transfer_id: int = 0

    _current_source = 0
    _current_destination = 0
    _current_repetition = 0
    _current_source_stride = 0
    _current_dest_stride = 0
    _in_flight: list[_InFlightDMA]

    def __init__(self):
        super().__init__()
        self._in_flight = []

    @staticmethod
    def _in_flight_to_event(end_observed: int, transfers: list[_InFlightDMA]):
        result = []
        for t in transfers:
            result.append(
                DMAEvent(t.started, end_observed - t.started, t.source, t.destination, t.inner_loop, t.transfer_id,
                         t.is_2d, t.source_strides, t.dest_strides, t.outer_loop))
        return result

    def cycle(self, state: TraceState) -> list[StreamingEvent]:
        result = []

        instruction = state.instruction
        if isinstance(instruction, DMSRCInstruction):
            self._current_source = instruction.read_source(state)
        elif isinstance(instruction, DMDSTInstruction):
            self._current_destination = instruction.read_destination(state)
        elif isinstance(instruction, DMREPInstruction):
            self._current_repetition = instruction.read_reps(state)
        elif isinstance(instruction, DMSTRInstruction):
            self._current_source_stride = instruction.read_source_strides(state)
            self._current_dest_stride = instruction.read_dest_strides(state)
        elif isinstance(instruction, DMCPYIInstruction):
            if instruction.is_2d:
                o = self._InFlightDMA(state.clock_cycle, self._current_source, self._current_destination,
                                      instruction.read_size(state), is_2d=True,
                                      source_strides=self._current_source_stride,
                                      dest_strides=self._current_dest_stride,
                                      outer_loop=self._current_repetition)
            else:
                o = self._InFlightDMA(state.clock_cycle, self._current_source, self._current_destination,
                                      instruction.read_size(state), is_2d=False)

            def write_back(txid):
                o.transfer_id = txid
                return []

            result += self.schedule_writeback(state, write_back)
            self._in_flight.append(o)

        elif isinstance(instruction, DMSTATIInstruction):
            if instruction.status == 2:
                def write_back(is_busy):
                    inner_result = []
                    if not is_busy:
                        inner_result += self._in_flight_to_event(state.clock_cycle, self._in_flight)
                        self._in_flight.clear()
                    return inner_result

                result += self.schedule_writeback(state, write_back)
            elif instruction.status == 0:
                def write_back(last_completed):
                    inner_result = []

                    # Turn all in-flight transfers that have been completed into events.
                    index = -1
                    for i, flight in enumerate(self._in_flight):
                        if flight.transfer_id > last_completed:
                            break
                        index = i

                    inner_result += self._in_flight_to_event(state.clock_cycle, self._in_flight[:index + 1])
                    del self._in_flight[:index + 1]
                    return inner_result

                result += self.schedule_writeback(state, write_back)

        return result


def worker(file: str):
    events = []
    generators = [BarrierEventGenerator(), StreamingEventGenerator(), DMAEventGenerator()]
    with open(file, 'r') as f:
        for index, l in enumerate(f):
            state = get_trace_state(l)
            if state is None:
                raise RuntimeError('Failed to parse trace: ' + l)

            for g in generators:
                if not state.stalled:
                    events += g.cycle(state)
                events += g.check_accumulator(state)
    return events


def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--inputs',
        metavar='<inputs>',
        type=argparse.FileType('rb'),
        nargs='+',
        help='Input performance metric dumps')
    parser.add_argument(
        '--traces',
        metavar='<trace>',
        nargs='*',
        help='Simulation traces to process')
    parser.add_argument(
        '--elf',
        nargs='?',
        help='ELF from which the traces were generated')
    parser.add_argument(
        '--addr2line',
        nargs='?',
        help='llvm-addr2line from quidditch toolchain')
    parser.add_argument(
        '-o',
        '--output',
        metavar='<json>',
        type=argparse.FileType('w'),
        nargs='?',
        default='events.json',
        help='Output JSON file')
    args = parser.parse_args()

    executor = ProcessPoolExecutor()
    futures = []
    for hartid, file in enumerate(args.traces):
        futures.append(executor.submit(worker, file))

    events = calculate_sections(args.inputs, args.elf, args.addr2line, args.traces)

    for hartid, f in enumerate(futures):
        events += map(lambda e: e.to_chrome_tracing(hartid), f.result())

    # Create TraceViewer JSON object
    json.dump({'traceEvents': events, 'displayTimeUnit': "ns"}, args.output, indent=2)


if __name__ == '__main__':
    sys.exit(main())
