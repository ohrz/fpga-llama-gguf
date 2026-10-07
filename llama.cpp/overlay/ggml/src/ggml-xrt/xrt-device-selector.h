// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <charconv>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

// XRT accepts either a numeric device index or a PCI BDF.  Keep the parsing
// strict here so an empty, malformed, or otherwise ambiguous selector cannot
// silently fall back to another device.
struct xrt_device_selector {
    bool is_index = true;
    unsigned int index = 0;
    std::string bdf;
};

inline bool xrt_device_selector_hex(char value) {
    return (value >= '0' && value <= '9') ||
           (value >= 'a' && value <= 'f') ||
           (value >= 'A' && value <= 'F');
}

inline bool xrt_device_selector_is_bdf(const std::string & value) {
    // domain:bus:slot.function, e.g. 0000:01:00.1
    return value.size() == 12 && value[4] == ':' && value[7] == ':' && value[10] == '.' &&
           xrt_device_selector_hex(value[0]) && xrt_device_selector_hex(value[1]) &&
           xrt_device_selector_hex(value[2]) && xrt_device_selector_hex(value[3]) &&
           xrt_device_selector_hex(value[5]) && xrt_device_selector_hex(value[6]) &&
           xrt_device_selector_hex(value[8]) && xrt_device_selector_hex(value[9]) &&
           xrt_device_selector_hex(value[11]);
}

inline xrt_device_selector parse_xrt_device_selector(const std::string & value) {
    if (value.empty()) throw std::invalid_argument("XRT device selector must not be empty");
    for (const char character : value) {
        if (std::isspace(static_cast<unsigned char>(character))) {
            throw std::invalid_argument("XRT device selector must not contain whitespace");
        }
    }

    bool numeric = true;
    for (const char character : value) numeric &= character >= '0' && character <= '9';
    if (numeric) {
        unsigned long long parsed = 0;
        const auto conversion = std::from_chars(value.data(), value.data() + value.size(), parsed, 10);
        if (conversion.ec != std::errc() || conversion.ptr != value.data() + value.size() ||
            parsed > std::numeric_limits<unsigned int>::max()) {
            throw std::invalid_argument("XRT device index is invalid: " + value);
        }
        return {true, static_cast<unsigned int>(parsed), {}};
    }

    if (!xrt_device_selector_is_bdf(value)) {
        throw std::invalid_argument("XRT device selector must be a numeric index or full PCI BDF: " + value);
    }
    return {false, 0, value};
}

inline xrt_device_selector xrt_device_selector_from_env(const char * name) {
    const char * value = std::getenv(name);
    // The default is part of the existing XRT interface.  It is still parsed
    // through the same strict path rather than selecting a device in a call
    // site with xrt::device(0).
    return parse_xrt_device_selector(value == nullptr ? "0" : value);
}
