#!/usr/bin/env python3

import argparse
import json
import sys


def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--inputs',
        metavar='<inputs>',
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
        '-o',
        '--output',
        metavar='<json>',
        nargs='?',
        default='events.json',
        help='Output JSON file')
    args = parser.parse_args()

    # TraceViewer events
    events = []
    for hartid, file in enumerate(args.inputs):
        with open(file, 'rb') as f:
            j = json.load(f)
            for index, section in enumerate(j):
                if index % 2 == 0:
                    continue

                start = section['tstart'] / 1e3
                dur = (section['tend'] - section['tstart']) / 1e3
                events.append({
                    # TODO: Get and demangle kernel name.
                    'name': f'section {index}',
                    'ts': start,
                    'dur': dur,
                    'ph': 'X',
                    'cat': 'kernel',
                    'pid': 0,
                    'tid': hartid,
                    'args': section,
                })


    # Create TraceViewer JSON object
    tvobj = {}
    tvobj['traceEvents'] = events
    tvobj['displayTimeUnit'] = "ns"

    # Dump TraceViewer events to JSON file
    with open(args.output, 'w') as f:
        json.dump(tvobj, f, indent=4)


if __name__ == '__main__':
    sys.exit(main())
