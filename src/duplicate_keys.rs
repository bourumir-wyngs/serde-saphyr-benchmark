// tests/duplicate_keys.rs
// Run with: `cargo test -- --nocapture`
//
// This test tries to parse a tiny YAML document with a duplicate key using each crate,
// and prints whether the crate REJECTS duplicates (error) or ACCEPTS them (last-wins / first-wins).
//
// Top-level duplicate keys are the simplest way to exercise the behavior.

use std::collections::BTreeMap;

// The smallest repro: duplicate key at the top level.
const DUP_YAML: &str = r#"
k: 1
k: 2
"#;

// A simple alias for the target type (mapping string -> i32).
type Map = BTreeMap<String, i32>;

fn mark(rejected: bool) -> &'static str {
    if rejected { "✅ reject" } else { "❌ accepts" }
}

pub fn duplicate_keys_across_crates() {
    println!("=== Duplicate key rejection  ===");
    // serde-saphyr
    let saphyr_rejects = serde_saphyr::from_str::<Map>(DUP_YAML).is_err();

    // serde-yaml-bw
    let yaml_bw_rejects = serde_yaml_bw::from_str::<Map>(DUP_YAML).is_err();

    // serde_yaml (deprecated)
    // Direct typed parse:
    let yaml_direct_rejects = serde_yaml::from_str::<Map>(DUP_YAML).is_err();
    // Value path (some crates differ here). Convert Value -> Map afterwards:
    let yaml_value_rejects = {
        match serde_yaml::from_str::<serde_yaml::Value>(DUP_YAML) {
            Ok(v) => Map::deserialize(v).is_err(),
            Err(_) => true,
        }
    };

    // serde_yaml_ng
    let yaml_ng_direct_rejects = serde_yaml_ng::from_str::<Map>(DUP_YAML).is_err();
    let yaml_ng_value_rejects = {
        match serde_yaml_ng::from_str::<serde_yaml_ng::Value>(DUP_YAML) {
            Ok(v) => Map::deserialize(v).is_err(),
            Err(_) => true,
        }
    };

    // serde_yml
    let yml_direct_rejects = serde_yml::from_str::<Map>(DUP_YAML).is_err();
    let yml_value_rejects = {
        match serde_yml::from_str::<serde_yml::Value>(DUP_YAML) {
            Ok(v) => Map::deserialize(v).is_err(),
            Err(_) => true,
        }
    };

    // serde_norway
    let norway_direct_rejects = serde_norway::from_str::<Map>(DUP_YAML).is_err();
    let norway_value_rejects = {
        match serde_norway::from_str::<serde_norway::Value>(DUP_YAML) {
            Ok(v) => Map::deserialize(v).is_err(),
            Err(_) => true,
        }
    };

    // Print a concise Markdown table (visible in test output with --nocapture).
    println!("| Crate           | Typed parse | Value -> Map |");
    println!("|----------------:|:------------|:-------------|");
    println!("| serde-saphyr    | {} | {} |", mark(saphyr_rejects), "—");
    println!("| serde-yaml-bw   | {} | {} |", mark(yaml_bw_rejects), "—");
    println!("| serde_yaml      | {} | {} |", mark(yaml_direct_rejects), mark(yaml_value_rejects));
    println!("| serde_yaml_ng   | {} | {} |", mark(yaml_ng_direct_rejects), mark(yaml_ng_value_rejects));
    println!("| serde_yml       | {} | {} |", mark(yml_direct_rejects), mark(yml_value_rejects));
    println!("| serde_norway    | {} | {} |", mark(norway_direct_rejects), mark(norway_value_rejects));

    // Optional: If you want to force a failure when *none* reject duplicates, uncomment:
    // assert!(
    //     saphyr_rejects || yaml_bw_rejects || yaml_direct_rejects || yaml_value_rejects ||
    //     yaml_ng_direct_rejects || yaml_ng_value_rejects || yml_direct_rejects || yml_value_rejects ||
    //     norway_direct_rejects || norway_value_rejects,
    //     "No crate rejected duplicate keys"
    // );
}

// Needed imports for Value -> Map conversions via Serde.
use serde::Deserialize;
use serde::{self};

