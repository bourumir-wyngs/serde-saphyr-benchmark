# Example: work directly on your ZIP and write to ./figures
python plot_criterion.py target/criterion.zip --group yaml_parse --outdir ./figures --baseline serde-yml

# Only make boxplots for a single size (e.g., 10MiB), skip the extra charts:
python plot_criterion.py target/criterion.zip --group yaml_parse --sizes 10MiB --no-median-line --no-relative

# If the distributions are very skewed, try log Y axis on the boxplots:
python plot_criterion.py target/criterion.zip --group yaml_parse --box-log-y
