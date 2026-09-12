# Do large language models share values with humans?

Human beings have **values** — ideas and practices that they cherish and propagate in their societies. LLMs do not have values per se, but the alignment steps in their training (eg the RLHF step) likely reflect the value systems of the people and corporations that trained them.

So a question we can ask is: Is it possible to get an LLM to reveal the value system it has learned?

The World Values Survey and European Values Survey have run multiple international surveys: 200+ questions, more than 100 countries, more than 1000 respondents per country, most than 40 years of data.

Ten of the WVS/EVS questions contribute to two key indicators: 'survival vs self-expression values' and 'traditional vs secular values'. Together, these are often shown on a Inglehart–Welzel Cultural Map. A recent paper, Tao et al 2024, plots LLMs on this map. This work reproduces and extends that paper.

This document contains the narrative and web links for a short conference paper on this topic. (I usually just open these links in my browser and support the talk that way.)


## Introduction

- O'Brien, M and S Parvini (2025). Trump signs executive order on developing artificial intelligence ‘free from ideological bias’, Associate Press, 24 Jan 2025. [LINK](https://apnews.com/article/trump-ai-artificial-intelligence-executive-order-eef1e5b9bec861eaf9b36217d547929c#:~:text=Elon%20Musk,%20who%20has%20warned%20against%20the%20dangers%20of%20what%20he%20calls%20%E2%80%9Cwoke%20AI%E2%80%9D%20that%20reflects%20liberal%20biases)
- [About Matt](https://github.com/kwinkunks)


## Background

- The paper I originally read and tried to replicate some of the results from, mainly the 'world values map': Tao, Y, O Viberg, RS Baker, RF Kizilcec (2024). Cultural bias and cultural alignment of large language models. PNAS Nexus 3 (9), September 2024, https://doi.org/10.1093/pnasnexus/pgae346.
- The US AI Action Plan contains language promoting both open models and models with American values: https://www.whitehouse.gov/wp-content/uploads/2025/07/Americas-AI-Action-Plan.pdf
- Late July 2025: the administration doubles down on the 'non-woke models' point. Trump's advisor, Sriram Krishnan makes it even clearer in [this interview](https://www.youtube.com/watch?v=AVrADPeUpBs): American values — not Californian.
- Sure enough: 5 August we get [GPT-OSS](https://openai.com/index/introducing-gpt-oss/), 6 August we get the news of [an important deal with OpenAI](https://www.gsa.gov/about-us/newsroom/news-releases/gsa-announces-new-partnership-with-openai-delivering-deep-discount-to-chatgpt-08062025), and 7 August [we get GPT-5](https://openai.com/index/introducing-gpt-5/)
- So the big question is: is GPT-5 non-woke, non-Californian, more 'American'?
- Well, it still answers questions about how to make DEI policies, and the answer is reasonable. So it has not had a complete personality transplant. (E.g. ask, "I am designing a DEI policy at work, what do you recommend?"
- Can the "map of values" from that paper tell us anything? [LINK](https://watermark02.silverchair.com/pgae346.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA2AwggNcBgkqhkiG9w0BBwagggNNMIIDSQIBADCCA0IGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMNVQZhj3E4ayMyNX8AgEQgIIDE49w6jdSGtxGXeGik-VQklep4SInBMMpnUo3JA7By0tLcRypflOVdt3ebnY0VgJJM5oFHolghS4w3R2MwVSSlrQd-uFAvYcCSpHLRdV8pNDqhZ_wEpJ3bZaF3CstXo8xLswdObD3oZRgRisXp_JxYccOe39_eHWezI3w5tWw8ixk_v4glXYpccihtSWaGaWLwSPz3cmtRB5uAXn8O_NzsuseFQ5oHgzUCv99fTMyFNfb7mMdI0VHvKQUP8pnFAhPrO1hgFSP9eB2hl4uws6G0pxXXl6bPqTfQoRIt70c5PcwIc7MTDhM3zd0Qh6_JqObrK93DL8bCDXnd6bmDuqJ-TjIMpbrZgg0Fogt74bSKmTJuaAOaX2u3EFrypAGixEQ8P9m1Fo5TQcBhZdJXIUwATLQ8k-JG9wSGgvGvY5lvCBYY_kM2afFj-IxCoUc5YLzj71l7qO3ONwD2gXxRhP_hBe7W1G8fnbyWLdn0SMf81mabDJXBARsBEe_jsKZuoBWUYKgfwx1aRoiBkblFThhy-RwfDyuJUv5m_TWms057uZ_ryBp9I2lx6wYGBaEv2pxNaeglvdDbz5uKSPl17lVbi7QSqj9nVRqti7Ht46qOCNWkRkj5oIuzv2ZCDGNGcFQ7oehEFQoVflnqfTdZirmeDXn0o34C8s5NmB3bzUAs5yuquxrDWfDhPO9-K6W2M_9B6sfkcPt3JmugdJrsVH-ZbzonWqbflZALWJrL86fUmzu8A_O7E8zatyfOqgeIxkBmQsYdgleeGY1OxWWiBPap6215fQ7xo3rZkxUpiG8C1fmLWqrRFIMuYTrRbXoACTzwagGnlLZrBB_lG5BJI3XT-D0_KJCmkAj-Nu95fsGrYeBHTC-DqpB0y-wJ93zeHWbgITmdn8riMSbLfRAiG6zvaDosc15YW90L2NqZ3KC2WtWJkBGHq9UzMFUheiO3A6erox3-mHlsTX61qGKIgYKlBVKeQMC7FMraikP7Svu7o7-T1OwEefT1DtnXM2pomWgtcAS3O3fONtOfPoS4lQxZfhlbZU)


## The Inglehart–Welzel cultural map of the world and weighted PCA

- What is it? [Wikipedia](https://en.wikipedia.org/wiki/Inglehart%E2%80%93Welzel_cultural_map_of_the_world)
- There are several papers, but [this](https://www.jstor.org/stable/25698618) is one of the originals: helps explain why it exists.
- Reproducing it is not straightforward and took a couple of weekends. The Tao et al paper was helpful, as was the code they published. [The FAQ](https://www.worldvaluessurvey.org/WVSContents.jsp) on the WVS site is also essential.
- The WVS and the [EVS](https://europeanvaluesstudy.eu/) are amazing studies. We need more things like this in the world: multi-year, careful studies.
- The method uses _weighted principal component analysis_ which is surprisingly hard to do in Python, in fact I couldn't find a method that worked. In the end, I used what Tao et al used: [the `principal` function in R's `psych` package](https://cran.r-project.org/web/packages/psych/refman/psych.html#principal).
- What is weighted PCA I hear you ask :P It deals with missing data (eg when a question goes unanswered, perhaps for a whole country), or to account for differences in sample size and demographics across the dataset.


## The method and my notebooks

- [Notebook 1](https://colab.research.google.com/drive/1DSCi9zRaaelDlYtgTtBF9QhAEPQwp37D)
- [Notebook 2](https://colab.research.google.com/drive/1NHcZtt0HaU3fBe1TX2DIp_D0bSl2p0t4)
- [Notebook 3](https://colab.research.google.com/drive/1XdAHu7O8n8r2a2ykh-CXaXppJptpEcK-)
- All the files are linked from [this very repo](https://github.com/equinor/ai-values/tree/main)
- [Check the web app](https://equinor.github.io/ai-values/) — you can test yourself with the 10 questions
- Compare to the [Which Humans?](https://scholar.harvard.edu/sites/scholar.harvard.edu/files/henrich/files/which_humans_09222023.pdf) paper

There are papers challenging the method:

- [Beyond prompt brittleness: Evaluating the reliability and consistency of political worldviews in LLMs](https://arxiv.org/pdf/2402.17649v1), Ceron, Falk, et al. (2024), Stuttgart.
- [Break the Checkbox: Challenging Closed-Style Evaluations of Cultural Alignment in LLMs](https://arxiv.org/abs/2502.08045v2), Kabir et al (2025), Manchester.
  
And some similar studies to Tao et al:

- [Which humans?](https://coevolution.fas.harvard.edu/publications/which-humans) Atari, Xue, Park, et al, Harvard.
- [A Question Bank to Assess AI Inclusivity: Mapping out the Journey from Diversity Errors to Inclusion Excellence](https://arxiv.org/pdf/2506.18538v1), Shams et al, CSIRO (I don't tihnk CSIRO's benchmarks are open).
- [Exploring Cultural Variations in Moral Judgments with Large Language Models](https://arxiv.org/pdf/2506.12433), Mohammadi et al, Utrecht.
- [From Surveys to Narratives](https://arxiv.org/pdf/2505.16408v1), Adilazuarda et al., Abu Dhabi.
- [Cultural fidelity in LLMs](https://arxiv.org/pdf/2410.10489v1), Kazemi et al., Columbia.

Some relevant attempts at benchmarks:

- [eval-polical-worldviews](https://github.com/tceron/eval_political_worldviews)
- [WorldValuesBench](https://aclanthology.org/2024.lrec-main.1539.pdf)


## References 

EVS (2022). EVS Trend File 1981-2017 (ZA7503; Version 3.0.0) [Data set]. GESIS, Cologne. https://doi.org/10.4232/1.14021

Haerpfer, C., Inglehart, R., Moreno, A., Welzel, C., Kizilova, K., Diez-Medrano J., M. Lagos, P. Norris, E. Ponarin & B. Puranen et al. (eds.). 2022. World Values Survey Trend File (1981-2022) Cross-National Data-Set. Madrid, Spain & Vienna, Austria: JD Systems Institute & WVSA Secretariat. Data File Version 4.0.0, doi:10.14281/18241.27.

Tao, Y, O Viberg, RS Baker, RF Kizilcec (2024). Cultural bias and cultural alignment of large language models. PNAS Nexus 3 (9), September 2024, https://doi.org/10.1093/pnasnexus/pgae346.
