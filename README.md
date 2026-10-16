# HD-EPIC real benchmark pilot

This workspace contains a compact, real-data HD-EPIC benchmark built from the official scene/object annotation metadata and selected raw video downloads.

## Project website (GitHub Pages)

The static project page is in [index.html](index.html) at the repository root. It includes a project overview, three real video examples with expandable answers, and links to the benchmark. No installation or build step is needed. Open the HTML file directly, or preview it from the repository root:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000`.

To publish the page using the current repository:

1. Commit and push `index.html`, `.nojekyll`, `docs/`, and this README to the `main` branch of `priyanka36/DEMO`.
2. Open the repository on GitHub and select **Settings → Pages**.
3. Under **Build and deployment**, select **Deploy from a branch**, choose **main** and **/(root)**, then **Save**.
4. Once GitHub finishes deployment, the project page will be at **https://priyanka36.github.io/DEMO/**.

These are the [GitHub Pages branch publishing instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site). To use `https://priyanka36.github.io/` without the project suffix, use a repository named `priyanka36.github.io` instead, copying `index.html`, `.nojekyll`, and `docs/` into its publishing root.

Edit the title and page content in `index.html`, and the styling in `docs/style.css`. Add the confirmed author list, affiliations, and paper link when available. If you rename the repository or change the default branch, update the GitHub links in the HTML.

The layout is inspired by the [GIRAFFE CVPR project page](https://m-niemeyer.github.io/project-pages/giraffe/index.html); the HTML and CSS are original. The six JPEGs in `docs/assets/` are 640-pixel frames extracted from the local HD-EPIC MP4s at the exact start/end timestamps of geometry question rows 24, 72, and 150 (zero-based). `docs/assets/examples.json` preserves these three source records, including their sequence IDs and evidence. The page displays rounded timestamps and credits the [HD-EPIC dataset](https://hd-epic.github.io/site/); the images remain subject to the source dataset’s terms. Update the displayed counts and examples if you regenerate the benchmark.

## Current benchmark
- Geometry benchmark output: [benchmark/geometry](benchmark/geometry)
- Affordance extension: [benchmark/affordance](benchmark/affordance)
- Raw data subset: [data/raw/HD-EPIC](data/raw/HD-EPIC)
- Processed reference files: [data/processed](data/processed)
- Summary table: [results/multi_video_question_summary.csv](results/multi_video_question_summary.csv)

## Project layout
- [data/raw/HD-EPIC](data/raw/HD-EPIC): downloaded raw HD-EPIC video and gaze files
- [data/processed](data/processed): processed object tracks and reference QA files
- [benchmark/geometry](benchmark/geometry): final multi-video geometry benchmark exports
- [benchmark/affordance](benchmark/affordance): affordance-aware extension built on top of the same tracks
- [results](results): benchmark summaries and reports

## Rebuild commands
```bash
cd /home/priyanka/VQA
python scripts/build_hd_epic_multivideo_pilot.py
python scripts/build_hd_epic_affordance_pilot.py
```

## LLM evaluation
The workspace also includes a benchmark-evaluation script that can score the question set against an LLM.

```bash
cd /home/priyanka/VQA
python scripts/evaluate_benchmark.py --provider dry-run --limit 5
python scripts/evaluate_benchmark.py --provider ollama --model llama3.1:8b
python scripts/evaluate_benchmark.py --provider openai --model gpt-4o-mini
```

The `dry-run` mode prints the formatted prompts without calling a model, which is useful for validating the benchmark format before an API call.

## Video-capable VLM suite
For the main benchmark, keep the small text-only models as baselines and evaluate a small set of actual video-capable VLMs on real MP4-backed rows.

```bash
cd /home/priyanka/VQA
python scripts/run_vlm_suite.py --config configs/vlm_models.json --start 0 --end 9 --valid-only --pull-missing --output-dir results/vlm_suite
```

This runner evaluates only rows whose `video_id` resolves to a real video in the HD-EPIC dataset, then writes one report per model under `results/vlm_suite/` plus a `suite_summary.json`.

## Affordance-aware extension
The project includes a geometry-first baseline and a higher-level affordance-aware extension built on the same real HD-EPIC object tracks.
- Baseline: [benchmark/geometry](benchmark/geometry)
- Extension: [benchmark/affordance](benchmark/affordance)

This extension keeps the original geometry benchmark intact and adds object-function style questions such as fixture compatibility, interaction readiness, and historical functional-state change.

## Notes
The official minimal subset is intentionally kept small and real. The project is designed for a symposium-ready pilot, not for full HD-EPIC corpus-scale download or training.
