from openrouter import OpenRouter
import os
import subprocess

OPENROUTER_API_KEY = open("LLM\ApiKey.md","r").read()

with OpenRouter(api_key=OPENROUTER_API_KEY) as client:

    agentmd = open("LLM\Agend.md","r",encoding="utf-8").read()
    skillmd = open("LLM\Skill.md","r",encoding="utf-8").read()
    messages = [{"role":"system","content":agentmd + skillmd}]    

    while True:
        user_input = input("\nPatch : ")
        messages.append({"role":"user", "content":user_input})

        print("\n-----------------Agent Start--------------")

        while True:
            respose = client.chat.send(                
                model="minimax/minimax-m2.5",                
                messages=messages
            )
  
            reply  = respose.choices[0].message.content
            messages.append({"role":"assistant", "content":reply})
    
            print(f" [AI] {reply}")

            if reply.strip().startswith("完成:"):
                print("\n-----------------Agent End--------------")
                print(f" [AI] {reply.strip().split('完成:')[1].strip()}")
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
            messages.append({"role":"user", "content":content})


  
  