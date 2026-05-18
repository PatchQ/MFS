"""
ScriptRunner - subprocess 調度器

使用 JSON-line 協議與 ProcessTA/ProcessAI 通信
"""

import subprocess
import json
import os
from typing import Iterator, List, Union, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .protocol import (
    TARequest, TAResponse,
    AIRequest, AIResponse,
    parse_json_line, emit_json_line
)


class ScriptRunner:
    """
    subprocess + JSON-line 協議調度器
    
    用法:
        runner = ScriptRunner("python ProcessTA.py --json-stdin")
        for response in runner.run_ta(requests):
            print(response)
    """

    def __init__(self, script_cmd: str, max_workers: int = 4):
        """
        初始化調度器
        
        Args:
            script_cmd: 完整命令，如 "python ProcessTA.py --json-stdin"
            max_workers: 最大并行任務數
        """
        self.script_cmd = script_cmd
        self.max_workers = max_workers

    def _run_subprocess(self, request: Union[TARequest, AIRequest]) -> str:
        """
        運行單個請求的 subprocess
        
        Args:
            request: TARequest 或 AIRequest
        
        Returns:
            JSON 響應字符串
        """
        process = subprocess.Popen(
            self.script_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        input_data = emit_json_line(request.to_dict())
        stdout, stderr = process.communicate(input=input_data)
        
        if process.returncode != 0 and stderr:
            raise RuntimeError(f"Process error: {stderr}")
        
        return stdout.strip()

    def run_single_ta(self, request: TARequest) -> TAResponse:
        """
        執行單個 TA 請求
        
        Args:
            request: TARequest 對象
        
        Returns:
            TAResponse 對象
        """
        output = self._run_subprocess(request)
        return TAResponse.from_dict(parse_json_line(output))

    def run_single_ai(self, request: AIRequest) -> AIResponse:
        """
        執行單個 AI 請求
        
        Args:
            request: AIRequest 對象
        
        Returns:
            AIResponse 對象
        """
        output = self._run_subprocess(request)
        return AIResponse.from_dict(parse_json_line(output))

    def run_ta(self, requests: List[TARequest]) -> Iterator[TAResponse]:
        """
        批量執行 TA 請求（并行）
        
        Args:
            requests: TARequest 列表
        
        Yields:
            TAResponse 對象
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.run_single_ta, req): req 
                for req in requests
            }
            for future in as_completed(futures):
                try:
                    yield future.result()
                except Exception as e:
                    req = futures[future]
                    yield TAResponse(
                        sno=req.sno,
                        stype=req.stype,
                        success=False,
                        error=str(e)
                    )

    def run_ai(self, requests: List[AIRequest]) -> Iterator[AIResponse]:
        """
        批量執行 AI 請求（并行）
        
        Args:
            requests: AIRequest 列表
        
        Yields:
            AIResponse 對象
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.run_single_ai, req): req 
                for req in requests
            }
            for future in as_completed(futures):
                try:
                    yield future.result()
                except Exception as e:
                    req = futures[future]
                    yield AIResponse(
                        sno=req.sno,
                        stype=req.stype,
                        model=req.model,
                        success=False,
                        error=str(e)
                    )


class ScriptRunnerSimple:
    """
    簡化版調度器 - 單線程順序執行
    
    用於調試或低並發場景
    """

    def __init__(self, script_cmd: str):
        self.script_cmd = script_cmd

    def run_ta(self, requests: List[TARequest]) -> List[TAResponse]:
        """順序執行 TA 請求"""
        results = []
        for req in requests:
            try:
                result = self.run_single_ta(req)
                results.append(result)
            except Exception as e:
                results.append(TAResponse(
                    sno=req.sno,
                    stype=req.stype,
                    success=False,
                    error=str(e)
                ))
        return results

    def run_ai(self, requests: List[AIRequest]) -> List[AIResponse]:
        """順序執行 AI 請求"""
        results = []
        for req in requests:
            try:
                result = self.run_single_ai(req)
                results.append(result)
            except Exception as e:
                results.append(AIResponse(
                    sno=req.sno,
                    stype=req.stype,
                    model=req.model,
                    success=False,
                    error=str(e)
                ))
        return results

    def run_single_ta(self, request: TARequest) -> TAResponse:
        """執行單個 TA 請求"""
        process = subprocess.Popen(
            self.script_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        input_data = emit_json_line(request.to_dict())
        stdout, stderr = process.communicate(input=input_data)
        
        if process.returncode != 0:
            raise RuntimeError(f"Process error: {stderr}")
        
        return TAResponse.from_dict(parse_json_line(stdout.strip()))

    def run_single_ai(self, request: AIRequest) -> AIResponse:
        """執行單個 AI 請求"""
        process = subprocess.Popen(
            self.script_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        input_data = emit_json_line(request.to_dict())
        stdout, stderr = process.communicate(input=input_data)
        
        if process.returncode != 0:
            raise RuntimeError(f"Process error: {stderr}")
        
        return AIResponse.from_dict(parse_json_line(stdout.strip()))
