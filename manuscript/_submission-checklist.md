<!--
Working note (leading underscore → Quarto ignores it). Distilled from AI & SOCIETY
"Instructions for Authors" (Springer Nature), captured from misc/Submission guidelines …pdf
on 2026-09-24. Verify against the live page before submitting — Springer revises these.
Each item is tagged by WHEN it applies:
  [QMD]       do in the .qmd source now / during writing
  [EXPORT]    apply at the final Word (.docx) export step
  [INTERFACE] submit via the online system, NOT in the manuscript (kept out for blinding)
  [ASSET]     concerns figures / data / code files
-->

# AI & SOCIETY — submission checklist

Journal: **AI & SOCIETY** (Springer Nature), hybrid OA. Follows **COPE**; **Springer Nature
research data policy**.

## 0. Double-blind review
- [QMD] **Remove all author-identifying info from the manuscript**: no author names,
  affiliations, emails, ORCID, funding, acknowledgements in the manuscript body or
  frontmatter. Our current frontmatter (Matt Hall / Equinor / mtha@equinor.com) must be
  stripped for the review build.
- [QMD] **Avoid identity-revealing self-citation** — cite own prior work in the third person;
  don't write "our previous work [Hall …]".
- [ASSET] Strip identifying info from **figures and supplementary files** too (author names,
  file metadata, repo URLs that reveal identity).
- [INTERFACE] Author names, affiliations, corresponding-author contact go **on a separate
  Title Page / in the submission interface**, not in the manuscript.
- [ASSET] If giving reviewers a **repo/data link**, anonymize authorship there too (e.g.
  anonymous.4open.science or an anonymized OSF/Zenodo link). Our GitHub repo reveals identity.
- **Suggested workflow:** keep one `hall-2026-llm-values.qmd` as the true source with author
  metadata, and produce an **anonymized review build** (author fields blanked, self-refs
  neutralized). Could be a Quarto param/profile so it's one flag, not a fork. Restore
  attribution only for the accepted version or preprints.

## 1. Article type & length
- Likely **Research Article**: "theoretically grounded, methodologically sound, empirically
  strong… significant contribution." **Normal length ~10,000 words.** [QMD] Keep body ≲10k.

## 2. Front matter
- [QMD] **Title**: concise and informative.
- [QMD] **Abstract**: **150–250 words**, no undefined abbreviations, no unspecified
  references. (May go up to 450 for original research if needed.) Current draft ≈190 — OK.
- [QMD] **Keywords**: **4–6**, for indexing.
- [INTERFACE] ORCID (recommended), corresponding-author email.

## 3. Headings, text formatting
- [QMD] **Decimal headings, max 3 levels** (1, 1.1, 1.1.1). We already use
  `number-sections`; just don't go past three levels.
- [QMD] Define abbreviations at first mention; use consistently.
- [QMD] **Footnotes, not endnotes**; numbered consecutively; a footnote must not be *only* a
  citation and must not carry full bib details.
- [EXPORT] Word specifics: plain font (they suggest 10-pt Times Roman), automatic page
  numbers, **no field functions**, tab stops (not spaces) for indents, **Word table
  function** for tables, **equation editor/MathType** for equations. (Pandoc/Quarto handles
  most of this; check no stray field codes.)

## 4. References — author–date (name–year)
- **Style = Springer Basic (author–date).** In-text by **name and year in parentheses**:
  "(Thompson 1990)", "Becker and Seligman (1996)", "(Abbott 1991; Barakat et al. 1995a, b)".
- [QMD] Set the CSL: `csl: springer-basic-author-date.csl` in frontmatter (get it from the
  citation-style-language repo). This changes our bracket-number look to name–year and
  formats the list to Springer style (initials, no ampersand, journal abbreviations, etc.).
- [QMD] Reference list **alphabetical by first author, then chronological**; only cited,
  published/accepted works. Personal communications/unpublished → in text only.
- [QMD] **Include DOIs as full links** (`https://doi.org/…`) — our `.bib` mostly has these;
  confirm all.
- Journal names: standard ISSN LTWA abbreviations (or full title if unsure).
- ⚠️ Check e.g. `@economist2026`, `@obrien2025trump`, `@zeff2023enter`, `@whitehouse2025actionplan`
  render acceptably as author–date (news/gov/dataset entries often need tidying).

## 5. Figures  [ASSET] + [QMD] captions
- [QMD] Cited in text in consecutive order, **Arabic numerals**; multi-part figures use
  lowercase **a, b, c**. Appendix figures continue main numbering.
- [QMD] **Caption style: begins "Fig. N" in bold, no punctuation after the number, none at
  the end of the caption**; caption text lives in the manuscript, not the image. Do NOT put a
  title inside the illustration. (Quarto defaults to "Figure 1." — adjust crossref/label
  prefix + reference-doc at export so it reads "Fig 1".)
- [ASSET] Resolution: **line art ≥1200 dpi; halftone ≥300 dpi; combination ≥600 dpi.** Our
  plots are matplotlib PNGs — **re-export the figure scripts at ≥600 dpi** (they're
  combination art: lines + text + color). Prefer vector (EPS/PDF) where possible; MS Office
  formats also accepted.
- [ASSET] **RGB, 8 bits/channel.** Color is free online; if any B&W print, ensure info
  survives greyscale and don't mention color in captions.
- [ASSET] **Accessibility**: use patterns *and* color (colorblind), caption describes the
  figure, lettering contrast ≥4.5:1, sans-serif (Helvetica/Arial) 8–12 pt, consistent sizes.
  → our cultural-map colors-by-region should get non-color redundancy.
- [ASSET] Width to column: large journal 84 mm (double-col) / 174 mm (single-col), height
  ≤234 mm. Name files `Fig1.…` etc. if uploaded separately (default: embed in the Word file).

## 6. Tables  [QMD]
- Arabic numerals, cited in order, each with a caption; table footnotes use superscript
  lowercase letters. Build with the table function (Quarto emits real Word tables).

## 7. Statements & Declarations  [INTERFACE] (+ separate title page)
Required for original research; **submission returned as incomplete without them.** Put under
a **"Declarations"** heading on the **separate title page / interface**, NOT in the blinded
manuscript:
- **Funding** (name funder + grant no., or an explicit "no funding" statement).
- **Competing interests** (financial & non-financial; last 3 yrs). Note: author is at
  Equinor — disclose employment.
- **Ethics approval / Consent** — n/a (no human subjects by *us*; WVS/EVS are secondary).
- **Data availability** — **mandatory.** Draft now (see §8).
- **Materials/Code availability**.
- **Author contributions** (free-text or CRediT taxonomy).

## 8. Data & code availability  [QMD Methods + INTERFACE]
- [QMD] **Data Availability Statement is mandatory.** Draft: the WVS & EVS microdata are
  **third-party, licensed, non-redistributable** (cite the datasets with DOIs — already in
  `.bib`); state where to obtain them (WVS/EVS/GESIS). The **derived artifacts and analysis
  code are openly available** (repo / archived release with DOI — Zenodo recommended).
- [QMD] Cite datasets in the reference list with persistent IDs (DataCite fields) — done.
- Consider depositing a **release with a DOI** (Zenodo) and citing it.

## 9. LLM-use disclosure  [QMD Methods] ⚠️ important for this paper
- LLMs **cannot be authors**. Any LLM use **beyond grammar/translation** must be
  **documented in the Methods** (or a suitable section). "AI-assisted copy editing"
  (readability/grammar/style, human-accountable) need NOT be declared, but **generative
  editorial work / content creation does.** The journal says such use is "strongly
  discouraged."
- Our README states the paper was made with "an even blend of human and AI contributions…
  new content such as text, images, analysis, and ideas." That **exceeds copy-editing** and
  **must be disclosed** — and be aware the journal's stance may affect desk-screening.
  Recommend an explicit, specific Methods paragraph on how Claude was used.

## 10. Ethics / integrity  [general]
- Not published/under review elsewhere; no salami-slicing; original; plagiarism-screened.
- Preprint/prior-posting must be acknowledged in declarations (if the repo/site counts).
- Avoid untrue statements about identifiable entities/people — relevant given the paper
  discusses named labs, Musk, Trump administration; keep claims sourced and measured.

## 11. Mechanics of the final Word export  [EXPORT]
- **Word (.docx) only** for this journal — go docx via `quarto render … --to docx` (verified
  working).
- Provide **editable source files** at every submission/revision (the .docx counts).
- Two builds to produce: **(a) anonymized manuscript .docx** (body + figures + captions +
  data/LLM statements that don't reveal identity), **(b) title page** (title, authors,
  affiliations, corresponding author, Declarations, acknowledgements).
- Add to frontmatter when ready:
  ```yaml
  format:
    docx:
      reference-doc: springer-template.docx   # optional house styling
  csl: springer-basic-author-date.csl
  ```

## Fastest path to "apply when needed"
1. Add `csl: springer-basic-author-date.csl` + a `docx` format block to the qmd. (quick)
2. Add a Quarto **profile/param for the anonymized review build** (blank author, neutralize
   self-refs). (quick)
3. Re-export figures at ≥600 dpi RGB with colorblind-safe redundancy + "Fig N" captions.
4. Draft the **Data Availability** + **LLM-use** + **Competing interests/Funding** statements.
5. Confirm abstract 150–250 w, keywords 4–6, headings ≤3 levels, body ≲10k, DOIs on all refs.
