from openrouter import OpenRouter
from flask import Flask, request, jsonify
import os
import subprocess

OPENROUTER_API_KEY = open("LLM\ApiKey.md","r").read()

app = Flask(__name__)
client = OpenRouter(api_key=OPENROUTER_API_KEY)
agentmd = open("LLM\Agend.md","r",encoding="utf-8").read()
skillmd = open("LLM\Skill.md","r",encoding="utf-8").read()
messages = [{"role":"system","content":agentmd + skillmd}]    
HTML = open("LLM\index.html","r",encoding="utf-8").read()

@app.get("/")
def index():
    return HTML

@app.post("/chat")
def chat():
    user_input = request.json["message"]
    print(f"[用戶]{user_input}")
    messages.append({"role":"user", "content":user_input})
    steps = []


    while True:
        response = client.chat.send(model="minimax/minimax-m2.5",messages=messages)           
        reply  = response.choices[0].message.content
        messages.append({"role":"assistant", "content":reply})
        print(f"\033[32m [AI] {reply}\033 [0m")
        steps.append({"type":"ai","content":reply})

        if reply.strip().startswith("完成:"):
            break

        command = reply.strip().split("命令:")[1].strip()
        command_result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,                # ← 等同 universal_newlines=True
            encoding="utf-8",
            errors="replace"
        ).stdout

        content = f"執行完畢 {command_result}"
        print(f" [Agent] {content}")
        steps.append({"type":"cmd","content":command_result})
        messages.append({"role":"user", "content":content})
    
    return jsonify(steps=steps)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8080)


  
  