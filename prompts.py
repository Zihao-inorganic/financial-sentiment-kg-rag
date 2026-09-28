"""Text merging, graph extraction and classification prompts (Appendices A–E)."""

import json

LABELS = {0: "Bearish", 1: "Bullish", 2: "Neutral"}


def merge(text_list):
    return f"""You are an AI assistant helping to merge financial texts. Please merge the
following multiple similar financial texts into one coherent article. While merging, please:
1. Remove duplicate content.
2. Resolve potential contradictions to ensure content consistency.
3. Add transitional sentences or use conjunctions to make the merged text more coherent.
Please return the result as a coherent English text.
List of texts:
{json.dumps(text_list, ensure_ascii=False, indent=4)}"""


def extract(merged_text, label):
    return f"""You are an AI assistant helping to build a knowledge graph from financial texts.
The sentiment label of the text is '{label}'.
Please extract a knowledge graph from the following text, focusing on the following perspectives:
1. **Numerical Context**: Identify any specific numbers or metrics mentioned and their
sentiment implications. Figures without context can be considered neutral.
2. **Temporal Prompting**: Extract any timeframes or specific contexts that might influence
the sentiment. If there is recent and past sentiment, the final sentiment should be based
on the most recent sentiment.
3. **Comparative Analysis**: If the statement compares performance with another entity or
timeframe, derive sentiment from this relative performance. For example, a decrease in
profitability represents negative sentiment; a growth in sales, increase in share price,
or reduction in loss is positive. When there is mixed sentiment, follow the rule that
improvement stands for positive in finance.
4. **Causal Attribution**: Identify and assess any causal factors or strategic moves
mentioned that carry sentiment.
5. **Risk and Uncertainty Analysis**: Evaluate any potential risks, threats, or
uncertainties that carry sentiment.
When extracting, please:
- Include the **distance** between entities in the `relations` as the number of hops
(e.g., direct relations have a distance of 1).
- Incorporate the sentiment label into the knowledge graph using relationship attributes,
entity attributes, and multiple relationship types.
- Ensure that the output is in **JSON format** with the following structure:
```json
{{
  "entities": [
    {{"name": "Entity Name", "type": "Entity Type",
      "attributes": {{"key1": "value1", "key2": "value2"}}}}
  ],
  "relations": [
    {{"source": "Source Entity Name", "target": "Target Entity Name",
      "type": "Relation Type", "sentiment": "Sentiment", "distance": 1}}
  ],
  "sentiment_label": "{label}"
}}
```
Important: Do not include any additional text or explanations. Only output the JSON data
in the specified format.
Text:
\"\"\"
{merged_text}
\"\"\""""


def sufficient(query, context):
    return f"""Given the following question:
"{query}"
And the following retrieved knowledge:
{context}
Is the retrieved knowledge sufficient to answer the question? Please answer 'Yes' or 'No'
and provide a brief justification."""


def answer(query, context=None):
    # Classify from the query when no retrieved context is supplied.
    prefix = (f"Use the following knowledge to answer the question:\nKnowledge:\n{context}\n"
              if context is not None else "")
    return prefix + f"""Question: {query}
Please determine the sentiment of the above content. The possible sentiment categories are:
0: Bearish
1: Bullish
2: Neutral
Please answer in the following format:
Sentiment: [0/1/2]"""


def cot(query):
    return f"""You are a financial sentiment analysis assistant.
Please analyze the sentiment of the given text step by step.
Follow this reasoning structure:
1. Numerical Context: Identify any specific numbers or metrics mentioned and their
sentiment implications.
2. Temporal Prompting: Extract timeframes or contexts that might influence sentiment.
If both recent and past sentiment appear, base the final judgment on the most recent one.
3. Comparative Analysis: Check for comparisons with other entities or timeframes.
Infer sentiment from relative performance (e.g., profit decrease = negative,
sales growth = positive). When mixed, improvements indicate positive sentiment.
4. Causal Attribution: Identify causal factors or strategic moves that imply sentiment.
5. Risk and Uncertainty Analysis: Evaluate risks, threats, or uncertainties that carry sentiment.
Finally, conclude with the sentiment:
0 = Bearish
1 = Bullish
2 = Neutral
Text: {query}
Please output your reasoning process first, then give the final sentiment in the format:
Sentiment: [0/1/2]."""
