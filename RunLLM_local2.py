from openai import OpenAI
import os
import subprocess

OPENROUTER_API_KEY="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"

with OpenAI(api_key=OPENROUTER_API_KEY,base_url="https://openrouter.ai/api/v1") as client:

    agentmd = open("Agend.md","r",encoding="utf-8").read()
    skillmd = open("Skill.md","r",encoding="utf-8").read()
    messages = [{"role":"system","content":agentmd + skillmd}]    

    while True:
        user_input = input("\nPatch : ")
        messages.append({"role":"user", "content":user_input})

        print("\n-----------------Agent Start--------------")

        while True:
            response = client.chat.completions.create(                
                    model="minimax/minimax-m2.5",
                    #model="openrouter/free",       
                    messages=messages,
                    temperature=0.7,
                    max_tokens=300
                )
  
            reply  = response.choices[0].message.content
            messages.append({"role":"assistant", "content":reply})
    
            print("本次 token 用量：", response.usage)
            print(f" [AI] {reply}")

            if reply.strip().startswith('###完成###:'):
                print("\n-----------------Agent End--------------")
                try:
                    print(f" [AI] {reply.strip().split('###完成###:')[1].strip()}")
                    break
                except IndexError:
                    print(f" [AI] {reply.strip()}")
                    break
            try:
                command = reply.strip().split('###命令###:')[1].strip()
                command_result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,                # ← 等同 universal_newlines=True
                encoding="utf-8",
                errors="replace"
            ).stdout     
            except IndexError:
                command_result = ""
                break


            content = f"###執行完畢### {command_result}"
            print(f" [Agent] {content}")
            messages.append({"role":"user", "content":content})


  
  