from openai import OpenAI
client = OpenAI(
  api_key="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"  # this is also the default, it can be omitted
)

# {'role': 'system', 'content': 'You are an assistant who follows instructions.'}
def prompt_llm(model, pre, prompt, temperature):
    response = client.chat.completions.create(
        model=model,
        messages=[
            pre,
            {'role': 'user', 'content': prompt}
        ],
        temperature=temperature
    )
    answer = response.choices[0].message.content
    return answer

if __name__ == '__main__':
    answer = prompt_llm("gpt-3.5-turbo", {'role': 'system', 'content': 'You are an assistant who follows instructions.'}, 'hello world', 0)
    print(answer)