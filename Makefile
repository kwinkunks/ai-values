# Makefile for the ai-values manuscript and figures.
#
#   make figures       regenerate paper figures as 600-dpi PNG rasters (draft)
#   make figures-eps   regenerate them as EPS vector art (production)
#   make blind         build the anonymised submission .docx (no author — for review)
#   make titlepage     build the separate title page .docx (authors + Declarations)
#   make preprint      build the .docx with authorship restored
#   make html          preview build (HTML, with authorship)
#   make clean         remove render artifacts
#
# Override resolution with e.g.  make figures FIG_DPI=300

MANUSCRIPT := manuscript/hall-2026-llm-values.qmd
TITLEPAGE  := manuscript/title-page.qmd
AUTHORS    := manuscript/_authors.yml
PY         := uv run python

FIG_DPI ?= 600
export FIG_DPI

.PHONY: figures figures-eps blind docx titlepage preprint html clean

figures:
	$(PY) figures/plot_cultural_map.py
	$(PY) figures/plot_variance.py
	$(PY) figures/plot_zeroshot.py
	$(PY) figures/plot_llms_lineage.py --lineage "Claude Opus"
	$(PY) figures/plot_llms_lineage.py --lineage "Claude Sonnet"
	# GPT spans several config lineages, so it needs an explicit model list.
	# Fill in the exact labels used in the paper, e.g.:
	# $(PY) figures/plot_llms_lineage.py --lineage GPT --out llms_gpt_all.$${FIG_EXT:-png} \
	#     --models "GPT-4" "GPT-4o" "GPT-5" "GPT-5.5" "GPT-6 Astra"

figures-eps: export FIG_EXT := eps
figures-eps: figures

# Blind = the submission build: author metadata is NOT in the .qmd, so this is anonymised.
blind docx:
	quarto render $(MANUSCRIPT) --to docx

# Separate title page (authors + Declarations), submitted alongside the blind manuscript.
titlepage:
	quarto render $(TITLEPAGE) --to docx

# Preprint/camera-ready = inject author metadata from _authors.yml.
preprint:
	quarto render $(MANUSCRIPT) --to docx --metadata-file $(AUTHORS)

html:
	quarto render $(MANUSCRIPT) --to html --metadata-file $(AUTHORS)

clean:
	rm -rf manuscript/*.docx manuscript/*.html manuscript/*_files
