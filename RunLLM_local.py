from openrouter import OpenRouter
import os
import subprocess

#OPENROUTER_API_KEY="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"
#OPENROUTER_API_KEY="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"
OPENROUTER_API_KEY = "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"

with OpenRouter(api_key=OPENROUTER_API_KEY) as client:

    agentmd = open("Agend.md","r",encoding="utf-8").read()
    skillmd = open("Skill.md","r",encoding="utf-8").read()
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


  
  