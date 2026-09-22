"""Check resource extraction against independent assembler and ELF fixtures."""

import struct
import unittest

from compile_resources import cubin_architecture, parse_ptxas_resources


class TestResourceExtraction(unittest.TestCase):
    def test_keep_each_kernel_stack_spill_and_register_counts(self):
        log = """ptxas info    : Compiling entry function 'closest' for 'sm_110'
ptxas info    : Function properties for closest
    320 bytes stack frame, 8 bytes spill stores, 4 bytes spill loads
ptxas info    : Used 69 registers, used 0 barriers, 368 bytes cmem[0]
ptxas info    : Compiling entry function 'anyhit' for 'sm_110'
ptxas info    : Function properties for anyhit
    320 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 63 registers, 368 bytes cmem[0]
"""
        self.assertEqual(
            parse_ptxas_resources(log),
            [
                {
                    "name": "closest",
                    "ptxas_target": "sm_110",
                    "registers": 69,
                    "stack_bytes": 320,
                    "spill_store_bytes": 8,
                    "spill_load_bytes": 4,
                },
                {
                    "name": "anyhit",
                    "ptxas_target": "sm_110",
                    "registers": 63,
                    "stack_bytes": 320,
                    "spill_store_bytes": 0,
                    "spill_load_bytes": 0,
                },
            ],
        )

    def test_missing_resource_lines_are_unknown_not_zero(self):
        self.assertEqual(
            parse_ptxas_resources("ptxas info : Compiling entry function 'ray' for 'sm_89'"),
            [
                {
                    "name": "ray",
                    "ptxas_target": "sm_89",
                    "registers": None,
                    "stack_bytes": None,
                    "spill_store_bytes": None,
                    "spill_load_bytes": None,
                }
            ],
        )

    def test_read_actual_sm_from_cuda_elf_flags(self):
        data = bytearray(64)
        data[:9] = b"\x7fELF\x02\x01\x01\x41\x07"
        struct.pack_into("<H", data, 18, 190)
        struct.pack_into("<I", data, 48, 0x006E056E)
        self.assertEqual(
            cubin_architecture(data),
            {"elf_class": 64, "elf_machine": 190, "elf_abi_version": 7, "elf_flags": "0x006e056e", "sm": 110},
        )

    def test_cuda_13_abi_moves_sm_into_second_flag_byte(self):
        data = bytearray(64)
        data[:9] = b"\x7fELF\x02\x01\x01\x41\x08"
        struct.pack_into("<H", data, 18, 190)
        struct.pack_into("<I", data, 48, 0x06006E02)
        self.assertEqual(cubin_architecture(data)["sm"], 110)

    def test_reject_non_cuda_elf_instead_of_reporting_architecture(self):
        with self.assertRaises(ValueError):
            cubin_architecture(b"not a cubin")
        data = bytearray(64)
        data[:7] = b"\x7fELF\x02\x01\x01"
        struct.pack_into("<H", data, 18, 62)
        with self.assertRaises(ValueError):
            cubin_architecture(data)


if __name__ == "__main__":
    unittest.main()
