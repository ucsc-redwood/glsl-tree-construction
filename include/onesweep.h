#pragma once
#define RADIX_BIN 256
#define RADIX_LOG 8
#define RADIX_BITS 8
#define RADIX_DIGITS 1 << RADIX_BITS
#define RADIX_PASS  (sizeof(unsigned int) * 8 + RADIX_BITS - 1) / RADIX_BITS

