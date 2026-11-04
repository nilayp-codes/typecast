# Label agreement (blind second pass vs gold)

Docs compared: 5 (jd_004, jd_005, jd_008, jd_014, jd_015). Gold = reference, blind = scored like model output.
With this few docs one disagreement moves a field by 20 points: read it as a sanity check, not a rate.

| field | agreement |
|---|---|
| title | 80.0% |
| company | 100.0% |
| seniority | 80.0% |
| mode | 80.0% |
| salary_mentioned | 100.0% |
| min_years_experience | 100.0% |
| city | 80.0% |
| country | 100.0% |
| skills_precision | 52.9% |
| skills_recall | 72.0% |
| skills_f1 | 61.0% |
| overall | 86.8% |

## Differences per doc

- jd_004: title; skills only in gold: fine-tuning techniques, hugging face transformers, multimodal ai; skills only in blind: apis, azure openai service, chain-of-thought, claude, few-shot learning, fine-tuning, gemini, generative ai, gpt-4, hugging face, large language models, llama, llm evaluation, prompt engineering, semantic search, vector databases
- jd_005: city; skills only in gold: agentic ai, aws, ci/cd, cosmos db, docker, git, google cloud platform, knowledge retrieval systems, kubernetes, microsoft azure; skills only in blind: azure openai, claude, flask, gemini, generative ai, llama, mistral, openai, rest apis, retrieval-augmented generation
- jd_008: skills only in gold: api integration, openai apis; skills only in blind: agentic ai, apis, data science, fine-tuning, large language models, openai api, retrieval-augmented generation, vector databases
- jd_014: skills only in gold: agents, data pipelines, evaluation framework, plugins; skills only in blind: agentic ai, apache kafka, arize phoenix, aws, claude code, codex, evaluation frameworks, google adk, langfuse, mcp
- jd_015: seniority; mode; skills only in gold: finance, machine learning engineering; skills only in blind: aws, machine learning, node.js, react native
