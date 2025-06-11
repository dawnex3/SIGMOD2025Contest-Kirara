#pragma once

#include "attribute.h"

namespace Contest::MorselLite {

inline constexpr DataType INVALID_DATA_TYPE = static_cast<DataType>(-1);
inline constexpr size_t INVALID_INDEX = ~static_cast<size_t>(0);
  
template<typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
inline constexpr T INVALID_INDEX_OF = ~static_cast<T>(0);

struct ColumnAttrDesc {
  DataType       data_type{INVALID_DATA_TYPE};
  size_t         index{INVALID_INDEX};
};

struct ProjectColumnDesc {
  size_t   ref_index{INVALID_INDEX};
  size_t   out_index{INVALID_INDEX};
  DataType data_type{INVALID_DATA_TYPE};
};

}