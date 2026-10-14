//! Macroblock and sub-macroblock types (Tables 7-11 to 7-14, 7-17, 7-18).

/// Prediction mode of a macroblock partition (MbPartPredMode / SubMbPredMode).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredMode {
    L0,
    L1,
    Bi,
    Direct,
}

impl PredMode {
    pub fn uses_l0(self) -> bool {
        matches!(self, Self::L0 | Self::Bi)
    }

    pub fn uses_l1(self) -> bool {
        matches!(self, Self::L1 | Self::Bi)
    }
}

/// Partitioning of an inter macroblock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Partition {
    P16x16,
    P16x8,
    P8x16,
    /// Four 8x8 sub-macroblocks, each with its own sub_mb_type.
    P8x8,
}

impl Partition {
    pub fn count(self) -> usize {
        match self {
            Self::P16x16 => 1,
            Self::P16x8 | Self::P8x16 => 2,
            Self::P8x8 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MbType {
    /// I_NxN: Intra_4x4, or Intra_8x8 when transform_size_8x8_flag is set.
    INxN,
    /// `I_16x16_<pred>_<cbp chroma>_<cbp luma 0/15>`.
    I16x16 {
        pred_mode: u8,
        cbp_chroma: u8,
        cbp_luma: u8,
    },
    IPcm,
    /// SI macroblock (Table 7-12): Intra_4x4 prediction with SI decoding.
    Si,
    /// P_Skip or B_Skip (inferred, never coded as mb_type).
    Skip,
    /// B_Direct_16x16.
    Direct16x16,
    Inter {
        partition: Partition,
        modes: [PredMode; 2],
    },
    /// P_8x8ref0: P_8x8 with all ref_idx_l0 inferred to be 0.
    P8x8Ref0,
}

impl MbType {
    pub fn is_intra(self) -> bool {
        matches!(
            self,
            Self::INxN | Self::I16x16 { .. } | Self::IPcm | Self::Si
        )
    }

    /// mb_type in an I slice (Table 7-11); values above 25 are invalid.
    pub fn from_i(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::INxN,
            1..=24 => {
                let k = v - 1;
                Self::I16x16 {
                    pred_mode: (k % 4) as u8,
                    cbp_chroma: (k / 4 % 3) as u8,
                    cbp_luma: if k >= 12 { 15 } else { 0 },
                }
            }
            25 => Self::IPcm,
            _ => return None,
        })
    }

    /// mb_type in an SI slice: 0 is SI, the rest follow Table 7-11.
    pub fn from_si(v: u32) -> Option<Self> {
        if v == 0 {
            Some(Self::Si)
        } else {
            Self::from_i(v - 1)
        }
    }

    /// mb_type in a P or SP slice (Table 7-13); 5..=30 are intra types.
    pub fn from_p(v: u32) -> Option<Self> {
        use PredMode::L0;
        Some(match v {
            0 => Self::Inter {
                partition: Partition::P16x16,
                modes: [L0, L0],
            },
            1 => Self::Inter {
                partition: Partition::P16x8,
                modes: [L0, L0],
            },
            2 => Self::Inter {
                partition: Partition::P8x16,
                modes: [L0, L0],
            },
            3 => Self::Inter {
                partition: Partition::P8x8,
                modes: [L0, L0],
            },
            4 => Self::P8x8Ref0,
            _ => return Self::from_i(v - 5),
        })
    }

    /// mb_type in a B slice (Table 7-14); 23..=48 are intra types.
    pub fn from_b(v: u32) -> Option<Self> {
        use PredMode::{Bi, L0, L1};
        // Rows 4..=21 pair two partition modes, alternating 16x8 and 8x16.
        const PAIRS: [[PredMode; 2]; 9] = [
            [L0, L0],
            [L1, L1],
            [L0, L1],
            [L1, L0],
            [L0, Bi],
            [L1, Bi],
            [Bi, L0],
            [Bi, L1],
            [Bi, Bi],
        ];
        Some(match v {
            0 => Self::Direct16x16,
            1 => Self::Inter {
                partition: Partition::P16x16,
                modes: [L0, L0],
            },
            2 => Self::Inter {
                partition: Partition::P16x16,
                modes: [L1, L1],
            },
            3 => Self::Inter {
                partition: Partition::P16x16,
                modes: [Bi, Bi],
            },
            4..=21 => {
                let k = (v - 4) as usize;
                let partition = if k % 2 == 0 {
                    Partition::P16x8
                } else {
                    Partition::P8x16
                };
                Self::Inter {
                    partition,
                    modes: PAIRS[k / 2],
                }
            }
            22 => Self::Inter {
                partition: Partition::P8x8,
                modes: [Bi, Bi],
            },
            _ => return Self::from_i(v - 23),
        })
    }
}

/// Sub-macroblock partitioning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubPartition {
    S8x8,
    S8x4,
    S4x8,
    S4x4,
}

impl SubPartition {
    pub fn count(self) -> usize {
        match self {
            Self::S8x8 => 1,
            Self::S8x4 | Self::S4x8 => 2,
            Self::S4x4 => 4,
        }
    }

    /// (width, height) in luma samples.
    pub fn size(self) -> (u8, u8) {
        match self {
            Self::S8x8 => (8, 8),
            Self::S8x4 => (8, 4),
            Self::S4x8 => (4, 8),
            Self::S4x4 => (4, 4),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubMbType {
    pub partition: SubPartition,
    pub mode: PredMode,
}

impl SubMbType {
    /// sub_mb_type in P macroblocks (Table 7-17).
    pub fn from_p(v: u32) -> Option<Self> {
        let partition = [
            SubPartition::S8x8,
            SubPartition::S8x4,
            SubPartition::S4x8,
            SubPartition::S4x4,
        ]
        .get(v as usize)?;
        Some(Self {
            partition: *partition,
            mode: PredMode::L0,
        })
    }

    /// sub_mb_type in B macroblocks (Table 7-18). B_Direct_8x8 is reported
    /// as 4x4 partitions as the table lists, for direct mode derivation.
    pub fn from_b(v: u32) -> Option<Self> {
        use PredMode::{Bi, Direct, L0, L1};
        use SubPartition::{S4x4, S4x8, S8x4, S8x8};
        let (partition, mode) = match v {
            0 => (S4x4, Direct),
            1 => (S8x8, L0),
            2 => (S8x8, L1),
            3 => (S8x8, Bi),
            4 => (S8x4, L0),
            5 => (S4x8, L0),
            6 => (S8x4, L1),
            7 => (S4x8, L1),
            8 => (S8x4, Bi),
            9 => (S4x8, Bi),
            10 => (S4x4, L0),
            11 => (S4x4, L1),
            12 => (S4x4, Bi),
            _ => return None,
        };
        Some(Self { partition, mode })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use PredMode::*;

    #[test]
    fn i_types_follow_table_7_11() {
        assert_eq!(MbType::from_i(0), Some(MbType::INxN));
        assert_eq!(
            MbType::from_i(1),
            Some(MbType::I16x16 {
                pred_mode: 0,
                cbp_chroma: 0,
                cbp_luma: 0
            })
        );
        assert_eq!(
            MbType::from_i(8),
            Some(MbType::I16x16 {
                pred_mode: 3,
                cbp_chroma: 1,
                cbp_luma: 0
            })
        );
        assert_eq!(
            MbType::from_i(13),
            Some(MbType::I16x16 {
                pred_mode: 0,
                cbp_chroma: 0,
                cbp_luma: 15
            })
        );
        assert_eq!(
            MbType::from_i(24),
            Some(MbType::I16x16 {
                pred_mode: 3,
                cbp_chroma: 2,
                cbp_luma: 15
            })
        );
        assert_eq!(MbType::from_i(25), Some(MbType::IPcm));
        assert_eq!(MbType::from_i(26), None);
    }

    #[test]
    fn b_types_follow_table_7_14() {
        let inter = |partition, a, b| {
            Some(MbType::Inter {
                partition,
                modes: [a, b],
            })
        };
        assert_eq!(MbType::from_b(5), inter(Partition::P8x16, L0, L0));
        assert_eq!(MbType::from_b(10), inter(Partition::P16x8, L1, L0));
        assert_eq!(MbType::from_b(15), inter(Partition::P8x16, L1, Bi));
        assert_eq!(MbType::from_b(16), inter(Partition::P16x8, Bi, L0));
        assert_eq!(MbType::from_b(21), inter(Partition::P8x16, Bi, Bi));
        assert_eq!(MbType::from_b(23), Some(MbType::INxN));
        assert_eq!(MbType::from_p(30), Some(MbType::IPcm));
        assert_eq!(MbType::from_p(31), None);
    }

    #[test]
    fn sub_types_follow_tables_7_17_and_7_18() {
        assert_eq!(
            SubMbType::from_p(2),
            Some(SubMbType {
                partition: SubPartition::S4x8,
                mode: L0
            })
        );
        assert_eq!(SubMbType::from_p(4), None);
        assert_eq!(
            SubMbType::from_b(9),
            Some(SubMbType {
                partition: SubPartition::S4x8,
                mode: Bi
            })
        );
        assert_eq!(
            SubMbType::from_b(0),
            Some(SubMbType {
                partition: SubPartition::S4x4,
                mode: Direct
            })
        );
        assert_eq!(SubMbType::from_b(13), None);
    }
}
