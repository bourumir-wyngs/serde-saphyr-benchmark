// src/main.rs
// Run with: `cargo run --quiet`
//
// Prints a Markdown table that summarizes per-crate support for:
// - Merge keys: "Native" (works out of the box), "⚠️" (works only via Value::apply_merge()), or "No" (fails both ways).
// - Nested enums (extra-nested): "✅"/"❌".
//
// Assumes serde_yml and serde_norway expose `Value` with `apply_merge()` like serde_yaml.

mod duplicate_keys;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Person {
    name: String,
    age: u32,
    city: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct People {
    alice: Person,
    bob: Person,
}

// -------- Nested enums (with one more level) --------
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Doc {
    status: Status,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum Status {
    Ok { payload: Mode },
    Error { code: i32 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum Mode {
    Full(Level),
    Partial,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum Level {
    Fast,
    Thorough,
}

const MERGE_KEYS_YAML: &str = r#"
defaults: &base
  name: "Unknown"
  age:  30
  city: "Zurich"

alice:
  <<: *base
  name: "Alice"

bob:
  <<: *base
  age:  35
"#;

const NESTED_ENUMS_YAML: &str = r#"
status:
  Ok:
    payload:
      Full: Thorough
"#;

const ERROR_YAML: &str = r#"
status:
  E: rror:
    code: 404
"#;

#[derive(Copy, Clone)]
enum MergeSupport {
    Native,
    ApplyMerge,
    No,
}

struct Row {
    name: &'static str,
    merge: MergeSupport,
    nested_enums: bool,
}

fn main() {
    let expected_people = People {
        alice: Person {
            name: "Alice".to_string(),
            age: 30,
            city: "Zurich".to_string(),
        },
        bob: Person {
            name: "Unknown".to_string(),
            age: 35,
            city: "Zurich".to_string(),
        },
    };
    let expected_doc = Doc {
        status: Status::Ok {
            payload: Mode::Full(Level::Thorough),
        },
    };

    let mut rows: Vec<Row> = Vec::new();

    // Helper for pattern repetition (no impl Trait in closure return type)
    fn works_direct<T>(
        f: impl Fn() -> Option<T>,
        expected: &T,
    ) -> bool
    where
        T: serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        f().map(|x| x == *expected).unwrap_or(false)
    }

    // serde-saphyr
    rows.push(Row {
        name: "serde-saphyr",
        merge: if works_direct(|| serde_saphyr::from_str(MERGE_KEYS_YAML).ok(), &expected_people) {
            MergeSupport::Native
        } else {
            MergeSupport::No
        },
        nested_enums: works_direct(|| serde_saphyr::from_str(NESTED_ENUMS_YAML).ok(), &expected_doc),
    });

    // serde-yaml-bw
    rows.push(Row {
        name: "serde-yaml-bw",
        merge: if works_direct(|| serde_yaml_bw::from_str(MERGE_KEYS_YAML).ok(), &expected_people) {
            MergeSupport::Native
        } else {
            MergeSupport::No
        },
        nested_enums: works_direct(|| serde_yaml_bw::from_str(NESTED_ENUMS_YAML).ok(), &expected_doc),
    });

    // serde_yaml
    let merge_yaml = if works_direct(|| serde_yaml::from_str(MERGE_KEYS_YAML).ok(), &expected_people)
    {
        MergeSupport::Native
    } else {
        // Value + apply_merge
        let via = (|| {
            let mut v: serde_yaml::Value = serde_yaml::from_str(MERGE_KEYS_YAML).ok()?;
            v.apply_merge().ok();
            People::deserialize(v).ok()
        })()
            .map(|x| x == expected_people)
            .unwrap_or(false);
        if via {
            MergeSupport::ApplyMerge
        } else {
            MergeSupport::No
        }
    };
    rows.push(Row {
        name: "serde_yaml",
        merge: merge_yaml,
        nested_enums: works_direct(|| serde_yaml::from_str(NESTED_ENUMS_YAML).ok(), &expected_doc),
    });

    // serde_yaml_ng
    let merge_yaml_ng =
        if works_direct(|| serde_yaml_ng::from_str(MERGE_KEYS_YAML).ok(), &expected_people) {
            MergeSupport::Native
        } else {
            let via = (|| {
                let mut v: serde_yaml_ng::Value = serde_yaml_ng::from_str(MERGE_KEYS_YAML).ok()?;
                v.apply_merge().ok();
                People::deserialize(v).ok()
            })()
                .map(|x| x == expected_people)
                .unwrap_or(false);
            if via {
                MergeSupport::ApplyMerge
            } else {
                MergeSupport::No
            }
        };
    rows.push(Row {
        name: "serde_yaml_ng",
        merge: merge_yaml_ng,
        nested_enums: works_direct(
            || serde_yaml_ng::from_str(NESTED_ENUMS_YAML).ok(),
            &expected_doc,
        ),
    });

    // serde_yml (assumed Value + apply_merge available)
    let merge_yml =
        if works_direct(|| serde_yml::from_str(MERGE_KEYS_YAML).ok(), &expected_people) {
            MergeSupport::Native
        } else {
            let via = (|| {
                let mut v: serde_yml::Value = serde_yml::from_str(MERGE_KEYS_YAML).ok()?;
                v.apply_merge().ok();
                People::deserialize(v).ok()
            })()
                .map(|x| x == expected_people)
                .unwrap_or(false);
            if via {
                MergeSupport::ApplyMerge
            } else {
                MergeSupport::No
            }
        };
    rows.push(Row {
        name: "serde_yml",
        merge: merge_yml,
        nested_enums: works_direct(|| serde_yml::from_str(NESTED_ENUMS_YAML).ok(), &expected_doc),
    });

    // serde_norway (assumed Value + apply_merge available)
    let merge_norway =
        if works_direct(|| serde_norway::from_str(MERGE_KEYS_YAML).ok(), &expected_people) {
            MergeSupport::Native
        } else {
            let via = (|| {
                let mut v: serde_norway::Value = serde_norway::from_str(MERGE_KEYS_YAML).ok()?;
                v.apply_merge().ok();
                People::deserialize(v).ok()
            })()
                .map(|x| x == expected_people)
                .unwrap_or(false);
            if via {
                MergeSupport::ApplyMerge
            } else {
                MergeSupport::No
            }
        };
    rows.push(Row {
        name: "serde_norway",
        merge: merge_norway,
        nested_enums: works_direct(
            || serde_norway::from_str(NESTED_ENUMS_YAML).ok(),
            &expected_doc,
        ),
    });

    // yaml-spanned (two-step process: parse to Value, then deserialize)
    let merge_yaml_spanned = if works_direct(
        || {
            let value = yaml_spanned::from_str(MERGE_KEYS_YAML).ok()?;
            yaml_spanned::from_value(&value.inner).ok()
        },
        &expected_people,
    ) {
        MergeSupport::Native
    } else {
        MergeSupport::No
    };
    rows.push(Row {
        name: "yaml-spanned",
        merge: merge_yaml_spanned,
        nested_enums: works_direct(
            || {
                let value = yaml_spanned::from_str(NESTED_ENUMS_YAML).ok()?;
                yaml_spanned::from_value(&value.inner).ok()
            },
            &expected_doc,
        ),
    });

    println!("Error");
    println!(
        "serde-saphyr: {e}",
        e = serde_saphyr::from_str::<Value>(ERROR_YAML).unwrap_err()
    );
    println!(
        "serde-yaml-bw: {e}",
        e = serde_yaml_bw::from_str::<Value>(ERROR_YAML).unwrap_err()
    );
    println!(
        "serde_yaml: {e}",
        e = serde_yaml::from_str::<Value>(ERROR_YAML).unwrap_err()
    );
    println!(
        "serde_yaml_ng: {e}",
        e = serde_yaml_ng::from_str::<Value>(ERROR_YAML).unwrap_err()
    );
    println!(
        "serde_yml: {e}",
        e = serde_yml::from_str::<Value>(ERROR_YAML).unwrap_err()
    );
    println!(
        "serde_norway: {e}",
        e = serde_norway::from_str::<Value>(ERROR_YAML).unwrap_err()
    );
    println!(
        "yaml_spanned: {e}",
        e = yaml_spanned::from_str(ERROR_YAML).unwrap_err()
    );

    // Print Markdown table
    println!("| Crate | Merge Keys | Nested Enums |");
    println!("|------:|:-----------|:-----------------------------|");
    for r in rows {
        let merge_str = match r.merge {
            MergeSupport::Native => "✅ Native",
            MergeSupport::ApplyMerge => "⚠️ apply_merge",
            MergeSupport::No => "❌",
        };
        let nested_str = if r.nested_enums { "✅" } else { "❌" };
        println!("| {} | {} | {} |", r.name, merge_str, nested_str);
    }

    duplicate_keys::duplicate_keys_across_crates();
}


