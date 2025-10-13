import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
# client = OpenAI(
#     organization=os.getenv("OPENAI_ORG"),
#     project=os.getenv("OPENAI_PROJECT"),
# )


# def get_observation(prompt):
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system",
#                  "content": "You are a data analyst helping a marketing intelligence team interpret promotional strategy data.  For each analysis you will provide up to 5 bullet points."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0,
#             max_tokens=3000,
#             top_p=1
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"⚠️ OpenAI API Error: {e}"
