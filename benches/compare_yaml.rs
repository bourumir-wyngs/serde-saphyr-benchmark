//! Compare YAML parsing throughput across crates with Criterion.
//!
//! Crates covered (enable/disable below by toggling the `use` lines):
//! - serde_yaml (baseline)
//! - serde_yaml_bw (serde-yaml-bw fork)
//! - serde_yaml_norway (serde-yaml-norway fork)
//! - serde_yml
//! - serde_saphyr (two regimes): budget=None, and budget=very large ("max")
//!
//! We generate one big YAML per target size and reuse it across all parsers,
//! timing only the parse step. Throughput is reported via Criterion's
//! `Throughput::Bytes`.

// --- Optional: fix allocator for more stable numbers ---
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::hint::black_box;
use std::time::Duration;

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use serde::Deserialize;

// --------------------- Types under test ---------------------

#[derive(Debug, Deserialize)]
struct Document {
    defaults: Defaults,
    items: Vec<Item>,
}

#[derive(Debug, Deserialize)]
struct Defaults {
    enabled: bool,
    roles: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Item {
    enabled: bool,
    roles: Vec<String>,
    id: usize,
    name: String,
    details: Details,
}

#[derive(Debug, Deserialize)]
struct Details {
    description: String,
    notes: Vec<String>,
}

// --------------------- Data generation ---------------------

/// Build a large YAML string roughly `target_size` bytes long.
///
/// # Parameters
/// - `target_size`: desired byte size target (approximate; generation stops when reached or exceeded)
/// - `notes_per_item`: how many notes per item to inflate size deterministically
///
/// # Returns
/// - `String` with YAML data
fn build_large_yaml(target_size: usize, notes_per_item: usize) -> String {
    let mut yaml = String::with_capacity(target_size + 1024);
    yaml.push_str("---\n");
    yaml.push_str("defaults:\n");
    yaml.push_str("  enabled: &defaults_enabled true\n");
    yaml.push_str("  roles: &defaults_roles\n");
    yaml.push_str("    - reader\n");
    yaml.push_str("    - writer\n");
    yaml.push_str("items:\n");

    let mut index = 0usize;
    while yaml.len() < target_size {
        let mut entry = format!(
            "  - enabled: *defaults_enabled\n    roles: *defaults_roles\n    id: {index}\n    name: item_{index:05}\n    details:\n      description: \"Item number {index:05} includes repeated notes for benchmarking performance.\"\n      notes:\n"
        );
        for note_index in 0..notes_per_item {
            entry.push_str(&format!(
                "        - \"Note {note_index:02} for item {index:05}. This is repeated content to enlarge the YAML payload size considerably.\"\n"
            ));
        }
        yaml.push_str(&entry);
        index += 1;
    }
    yaml
}

// --------------------- Parser adapters ---------------------

// NOTE: Each adapter exposes a uniform `fn parse_<crate>(yaml: &str) -> Document`
// and unwraps internally (benches should fail loudly if parsing fails).
// If your fork uses a different module path or function name, adjust the `use`
// and the call inside the adapter.

#[allow(dead_code)]
fn parse_serde_yaml(yaml: &str) -> Document {
    // Baseline serde_yaml
    use serde_yaml as SY;
    let doc: Document = SY::from_str(black_box(yaml)).expect("serde_yaml parse failed");
    black_box(doc)
}

#[allow(dead_code)]
fn parse_serde_yaml_ng(yaml: &str) -> Document {
    // Baseline serde_yaml
    use serde_yaml_ng as SY;
    let doc: Document = SY::from_str(black_box(yaml)).expect("serde_yaml_ng parse failed");
    black_box(doc)
}


#[allow(dead_code)]
fn parse_serde_yaml_bw(yaml: &str) -> Document {
    // Fork: serde-yaml-bw (assumed crate name serde_yaml_bw)
    use serde_yaml_bw as SYBW;
    let doc: Document = SYBW::from_str(black_box(yaml)).expect("serde_yaml_bw parse failed");
    black_box(doc)
}

#[allow(dead_code)]
fn parse_serde_yaml_norway(yaml: &str) -> Document {
    // Fork: serde-norway (assumed crate name serde_norway)
    use serde_norway as SYN;
    let doc: Document = SYN::from_str(black_box(yaml)).expect("serde_yaml_norway parse failed");
    black_box(doc)
}

#[allow(dead_code)]
fn parse_serde_yml(yaml: &str) -> Document {
    // Crate: serde_yml
    use serde_yml as SYML;
    let doc: Document = SYML::from_str(black_box(yaml)).expect("serde_yml parse failed");
    black_box(doc)
}

#[allow(dead_code)]
fn parse_saphyr_budget_none(yaml: &str) -> Document {
    use serde_saphyr::{Error, Options};
    // Budget off: pure parser cost without guard bookkeeping (faster, but YAML must be own and controlled)
    let opts = Options { budget: None, ..Options::default() };
    let doc: Result<Document, Error> = serde_saphyr::from_str_with_options(black_box(yaml), opts);
    let doc = doc.expect("serde_saphyr (budget=None) parse failed");
    black_box(doc)
}

#[allow(dead_code)]
fn parse_saphyr_budget_max(yaml: &str) -> Document {
    use serde_saphyr::{budget::Budget, Error, Options};

    // Default budget would trigger DOS protection. This has no much impact on actual
    // performance because we only define threshold values we compare against.
    let many: usize = usize::MAX / 4;
    let opts = Options {
        budget: Some(Budget {
            max_reader_input_bytes: Some(many),
            max_events: many,
            max_aliases: many,
            max_anchors: many,
            max_depth: many,
            max_documents: many,
            max_nodes: many,
            max_total_scalar_bytes: many,
            max_merge_keys: many,
            enforce_alias_anchor_ratio: false,
            alias_anchor_min_aliases: many,
            alias_anchor_ratio_multiplier: many,
        }),
        ..Options::default()
    };
    let doc: Result<Document, Error> = serde_saphyr::from_str_with_options(black_box(yaml), opts);
    let doc = doc.expect("serde_saphyr (budget=max) parse failed");
    black_box(doc)
}

// --------------------- Criterion bench ---------------------

/// Register all comparisons in a single Criterion group.
///
/// # Parameters
/// - `c`: Criterion context (provided by harness)
fn bench_compare_yaml(c: &mut Criterion) {
    // Sizes to sweep (MiB)
    let sizes_mib = [1usize, 5, 10, 25, 50];
    let notes_per_item = 20;

    // Group configuration: longer measurement for stability on larger inputs
    let mut group = c.benchmark_group("yaml_parse");
    group.sampling_mode(SamplingMode::Auto);
    group.warm_up_time(Duration::from_secs(30));
    group.measurement_time(Duration::from_secs(600));
    group.sample_size(64);

    for &mib in &sizes_mib {
        let target_bytes = mib * 1024 * 1024;
        let yaml = build_large_yaml(target_bytes, notes_per_item);
        let yaml_len = yaml.len() as u64;
        group.throughput(Throughput::Bytes(yaml_len));

        // ---- serde_yaml (baseline) ----
        group.bench_with_input(
            BenchmarkId::new("serde_yaml", format!("{}MiB", mib)),
            &yaml,
            |b, y| b.iter(|| { let doc = parse_serde_yaml(y); black_box(doc.items.len()); }),
        );

        // ---- serde_yaml (baseline) ----
        group.bench_with_input(
            BenchmarkId::new("serde_yaml_ng", format!("{}MiB", mib)),
            &yaml,
            |b, y| b.iter(|| { let doc = parse_serde_yaml_ng(y); black_box(doc.items.len()); }),
        );

        // ---- serde_yaml_bw (if you use it) ----
        // Comment out if not in Cargo.toml
        group.bench_with_input(
            BenchmarkId::new("serde_yaml_bw", format!("{}MiB", mib)),
            &yaml,
            |b, y| b.iter(|| { let doc = parse_serde_yaml_bw(y); black_box(doc.items.len()); }),
        );

        // ---- serde_yaml_norway (if you use it) ----
        group.bench_with_input(
            BenchmarkId::new("serde_yaml_norway", format!("{}MiB", mib)),
            &yaml,
            |b, y| b.iter(|| { let doc = parse_serde_yaml_norway(y); black_box(doc.items.len()); }),
        );

        // ---- serde_yml ----
        group.bench_with_input(
            BenchmarkId::new("serde_yml", format!("{}MiB", mib)),
            &yaml,
            |b, y| b.iter(|| { let doc = parse_serde_yml(y); black_box(doc.items.len()); }),
        );

        // ---- serde_saphyr (budget=None) ----
        group.bench_with_input(
            BenchmarkId::new("serde_saphyr/budget_none", format!("{}MiB", mib)),
            &yaml,
            |b, y| b.iter(|| { let doc = parse_saphyr_budget_none(y); black_box(doc.items.len()); }),
        );

        // ---- serde_saphyr (budget=max) ----
        group.bench_with_input(
            BenchmarkId::new("serde_saphyr/budget_max", format!("{}MiB", mib)),
            &yaml,
            |b, y| b.iter(|| { let doc = parse_saphyr_budget_max(y); black_box(doc.items.len()); }),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_compare_yaml);
criterion_main!(benches);
