use core::{
    cmp::{Eq, Ord, PartialEq, PartialOrd},
    fmt::Debug,
    ops::{Add, Div, Mul, Neg, Sub},
};

use crate::generated;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
}

impl Into<generated::tensor_buffers::DataType> for DataType {
    fn into(self) -> generated::tensor_buffers::DataType {
        match self {
            DataType::Int8 => generated::tensor_buffers::DataType::Int8,
            DataType::Int16 => generated::tensor_buffers::DataType::Int16,
            DataType::Int32 => generated::tensor_buffers::DataType::Int32,
            DataType::Int64 => generated::tensor_buffers::DataType::Int64,
            DataType::UInt8 => generated::tensor_buffers::DataType::UInt8,
            DataType::UInt16 => generated::tensor_buffers::DataType::UInt16,
            DataType::UInt32 => generated::tensor_buffers::DataType::UInt32,
            DataType::UInt64 => generated::tensor_buffers::DataType::UInt64,
            DataType::Float32 => generated::tensor_buffers::DataType::Float32,
            DataType::Float64 => generated::tensor_buffers::DataType::Float64,
        }
    }
}

// Local Zero trait
pub trait Zero: Sized + PartialEq + Copy {
    fn zero() -> Self;
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

// Local One trait
pub trait One: Sized + PartialEq + Copy {
    fn one() -> Self;
}

pub trait Num:
    Zero
    + One
    + Copy
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    fn data_type() -> DataType;
}

// Local Float trait
pub trait Float: Num + Neg<Output = Self> {
    fn nan() -> Self;
    fn infinity() -> Self;
    fn neg_infinity() -> Self;
    fn is_nan(&self) -> bool;
    fn is_infinite(&self) -> bool;
    fn is_finite(&self) -> bool;
    // Add other relevant float methods if needed (e.g., sqrt, abs, etc.)
}

// Signed Integer trait
pub trait Int: Num + Eq + Ord + Neg<Output = Self> {}

// Unsigned Integer trait
pub trait UInt: Num + Eq + Ord {}

macro_rules! impl_num_base {
    ($t:ty, $dt:expr) => {
        impl Num for $t {
            fn data_type() -> DataType {
                $dt
            }
        }
        impl Zero for $t {
            fn zero() -> Self {
                0 as $t
            }
        }
        impl One for $t {
            fn one() -> Self {
                1 as $t
            }
        }
    };
}

macro_rules! impl_signed_integer {
    ($t:ty, $dt:expr) => {
        impl_num_base!($t, $dt);
        impl Int for $t {}
    };
    ($($t:ty),*) => {
        $(
            impl_num_base!($t, DataType::Int64); // Placeholder, adjust DataType accordingly
            impl Int for $t {}
        )*
    };
}

macro_rules! impl_unsigned_integer {
     ($t:ty, $dt:expr) => {
        impl_num_base!($t, $dt);
        impl UInt for $t {}
    };
    ($($t:ty),*) => {
        $(
            impl_num_base!($t, DataType::UInt64); // Placeholder, adjust DataType accordingly
            impl UInt for $t {}
        )*
    };
}

macro_rules! impl_float {
    ($t:ty, $dt:expr) => {
        impl_num_base!($t, $dt);
        impl Float for $t {
            fn nan() -> Self {
                <$t>::NAN
            }
            fn infinity() -> Self {
                <$t>::INFINITY
            }
            fn neg_infinity() -> Self {
                <$t>::NEG_INFINITY
            }
            fn is_nan(&self) -> bool {
                <$t>::is_nan(*self)
            }
            fn is_infinite(&self) -> bool {
                <$t>::is_infinite(*self)
            }
            fn is_finite(&self) -> bool {
                <$t>::is_finite(*self)
            }
        }
    };
}

// Implementations
impl_signed_integer!(i8, DataType::Int8);
impl_signed_integer!(i16, DataType::Int16);
impl_signed_integer!(i32, DataType::Int32);
impl_signed_integer!(i64, DataType::Int64);

impl_unsigned_integer!(u8, DataType::UInt8);
impl_unsigned_integer!(u16, DataType::UInt16);
impl_unsigned_integer!(u32, DataType::UInt32);
impl_unsigned_integer!(u64, DataType::UInt64);

impl_float!(f32, DataType::Float32);
impl_float!(f64, DataType::Float64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signed_integers() {
        assert_eq!(i8::zero(), 0);
        assert_eq!(i16::one(), 1);
        assert_eq!(i32::data_type(), DataType::Int32);
    }

    #[test]
    fn test_unsigned_integers() {
        assert_eq!(u8::zero(), 0);
        assert_eq!(u16::one(), 1);
        assert_eq!(u32::data_type(), DataType::UInt32);
    }

    #[test]
    fn test_floats() {
        assert_eq!(f32::zero(), 0.0);
        assert_eq!(f64::one(), 1.0);
        assert_eq!(f32::data_type(), DataType::Float32);
        assert!(f64::nan().is_nan());
    }
}
