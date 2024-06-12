#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
import sys
import functools
import typing


class KernelNameResolver:
    elf: str
    addr2line: str
    traces: tuple[typing.IO]

    def __init__(self, elf, addr2line, traces):
        self.elf = elf
        self.addr2line = addr2line
        self.traces = traces

    @functools.cache
    def get_name_from_address(self, address: int):
        p = subprocess.run([self.addr2line, '-e', self.elf, '-f', hex(address)], check=True, capture_output=True,
                           text=True)
        return p.stdout.splitlines()[0]

    def get_name(self, cycle: int, hartid: int):
        pattern = re.compile(r"\s*[0-9]+ " + str(cycle))
        iterator = self.traces[hartid]
        for l in iterator:
            if pattern.match(l):
                break
        else:
            return "<unknown-kernel>"

        cycle_regex = re.compile(r"\s*[0-9]+ [0-9]+\s+[0-9]+\s+(0x[0-9a-f]+).*#;(.*)")
        for index, l in enumerate(iterator):
            # Give up.
            if index == 100:
                return "<unknown-kernel>"

            l: str
            match = cycle_regex.match(l)
            if match is None:
                return "<unknown-kernel>"

            pc = int(match.group(1), base=16)
            state = eval(match.group(2))
            if not state['stall'] and state['pc_d'] != pc + 4:
                return self.get_name_from_address(state['pc_d'])


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
        type=argparse.FileType('r'),
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

    resolver = KernelNameResolver(args.elf, args.addr2line, (*args.traces,))

    # TraceViewer events
    events = []
    for hartid, file in enumerate(args.inputs):
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

            start = section['tstart'] / 1e3
            dur = (section['tend'] - section['tstart']) / 1e3
            events.append({
                'name': name,
                'ts': start,
                'dur': dur,
                'ph': 'X',
                'cat': 'vm' if is_dm_core else 'kernel,' + origin,
                'pid': 0,
                'tid': hartid,
                'args': section,
            })

    # Create TraceViewer JSON object
    json.dump({'traceEvents': events, 'displayTimeUnit': "ns"}, args.output, indent=4)


if __name__ == '__main__':
    sys.exit(main())
