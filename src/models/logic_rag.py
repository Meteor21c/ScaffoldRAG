import json
import logging
import pdb
import time
from typing import List, Dict, Tuple, Any
from src.models.base_rag import BaseRAG
from src.utils.utils import get_response_with_retry, fix_json_response
from colorama import Fore, Style, init

# Initialize colorama
init()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class LogicRAG(BaseRAG):

    def __init__(self, corpus_path: str = None, cache_dir: str = "./cache", filter_repeats: bool = False):
        """Initialize the LogicRAG system."""
        super().__init__(corpus_path, cache_dir)
        self.max_rounds = 3  # Default max rounds for iterative retrieval
        self.MODEL_NAME = "LogicRAG"
        self.filter_repeats = filter_repeats  # Option to filter repeated chunks across rounds

    def set_max_rounds(self, max_rounds: int):
        """Set the maximum number of retrieval rounds."""
        self.max_rounds = max_rounds


    def process_step(self, global_question: str, sub_query: str, contexts: List[str]) -> Dict[str, str]:
        """
        处理单个推理步骤：对检索内容进行总结，并尝试回答子问题。

        Args:
            global_question: 用户最原始的问题 (用于保持上下文目标)
            sub_query: 当前步骤的子查询
            contexts: 当前步骤检索到的 top-k 文档

        Returns:
            Dict containing 'summary' and 'answer'
        """
        context_text = "\n".join(contexts)

        # 构建 Prompt，要求同时输出总结和直接答案
        # 这里使用了 JSON 格式输出，便于程序解析存储
        prompt = f"""
        You are an intelligent reasoning agent. 
        Global Goal: Answer the question "{global_question}"
        Current Step: Investigate the sub-query "{sub_query}"

        Retrieved Information:
        {context_text}

        Task:
        1. Summarize the retrieved information relevant to both the Global Goal and Current Step. Be concise but preserve key entities and facts.
        2. Attempt to give a direct answer to the "Current Step" sub-query based ONLY on the retrieved information.
        3.Format your response as a JSON object with two keys:
        - "summary": "The concise summary..."
        - "answer": "The direct answer to the sub-query..."
        4.Do not output any other content.
        """

        try:
            response = get_response_with_retry(prompt)
            result = fix_json_response(response)

            # 简单的错误处理，防止解析失败
            if not result or "summary" not in result:
                return {
                    "summary": context_text[:500] + "...",  # Fallback
                    "answer": "Could not parse answer."
                }
            return result

        except Exception as e:
            logger.error(f"{Fore.RED}Error in process_step: {e}{Style.RESET_ALL}")
            return {
                "summary": context_text,
                "answer": "Error during processing."
            }

    def warm_up_analysis(self, question: str, history: List[Dict]) -> Dict:
        """
                Warm-up analysis: Analyze if the initial retrieval (Step 0) is sufficient.

                Args:
                    question: The global question.
                    history: The structured history (containing the initial attempt).
                """
        # 1. 格式化历史信息 (复用我们定义的工具函数)
        history_text = self._format_history_for_llm(history)
        try:
            prompt = f"""
            
            Global Question: {question}

            Current Knowledge (from Initial Retrieval):
            {history_text}

Based on the Current Information provided, analyze:
1. Can the global question be answered completely ONLY with knowledge given above? (Yes/No)
2. What specific information is missing, if any?
3. What specific question should we ask to find the missing information?
4. Summarize our current understanding based on available information.
5. What are the key dependencies needed to answer this question?
6. Why is information missing? (max 20 words)

Please format your response as a JSON object with these keys:
- "can_answer": boolean
- "missing_info": string
- "subquery": string
- "current_understanding": string
- "dependencies": list of strings (key information dependencies)
- "missing_reason": string (brief explanation why info is missing, max 20 words)"""

            response = get_response_with_retry(prompt)

            # Clean up response to ensure it's valid JSON
            response = response.strip()

            # Remove any markdown code block markers
            response = response.replace('```json', '').replace('```', '')

            # Parse the cleaned response using fix_json_response
            result = fix_json_response(response)
            if result is None:
                return {
                    "can_answer": True,
                    "missing_info": "",
                    "subquery": question,
                    "current_understanding": "Failed to parse reflection response.",
                    "dependencies": ["Information relevant to the question"],
                    "missing_reason": "Parse error occurred"
                }

            # Validate required fields
            required_fields = ["can_answer", "missing_info", "subquery", "current_understanding"]
            if not all(field in result for field in required_fields):
                logger.error(f"{Fore.RED}Missing required fields in response: {response}{Style.RESET_ALL}")
                raise ValueError("Missing required fields")

            # Add default values for new interpretability fields if missing
            if "dependencies" not in result:
                result["dependencies"] = ["Information relevant to the question"]
            if "missing_reason" not in result:
                result["missing_reason"] = "Additional context needed" if not result[
                    "can_answer"] else "No missing information"

            # Ensure boolean type for can_answer
            result["can_answer"] = bool(result["can_answer"])

            # Ensure non-empty subquery
            if not result["subquery"]:
                result["subquery"] = question

            return result

        except Exception as e:
            logger.error(f"{Fore.RED}Error in analyze_dependency_graph: {e}{Style.RESET_ALL}")
            return {
                "can_answer": True,
                "missing_info": "",
                "subquery": question,
                "current_understanding": f"Error during analysis: {str(e)}",
                "dependencies": ["Information relevant to the question"],
                "missing_reason": "Analysis error occurred"
            }


    def dependency_aware_rag(self, question: str, history: List[Dict], dependencies: List[str], idx: int) -> Dict:
        """
        Analyze if the question can be answered given the structured history.
        """
        # 1. 格式化历史信息
        history_text = self._format_history_for_llm(history)

        try:
            prompt = f"""
             We are solving the question: "{question}" by breaking it down into dependencies.

             Current Reasoning Chain (Executed Steps):
             {history_text}

             Pending Dependencies (To be solved):
             {dependencies[idx:]}

             Current dependency to be answered next: {dependencies[idx]}

             Please analyze:
             1. Based on the "Current Reasoning Chain", can the original question ("{question}") be answered completely NOW? (Yes/No)
             2. Summarize our current understanding based on the chain.

             Format response as JSON:
             - "can_answer": boolean (true or false)
             - "current_understanding": string

             Attention:Do not output any other content.
             """
            response = get_response_with_retry(prompt)
            result = fix_json_response(response)
            # 【修复重点】增加空值检查
            if result is None:
                logger.warning(
                    f"{Fore.YELLOW}dependency_aware_rag received invalid JSON. Using fallback.{Style.RESET_ALL}")
                return {
                    "can_answer": False,  # 解析失败时保守起见认为不能回答，继续检索
                    "current_understanding": "Failed to parse dependency analysis response."
                }

            return result
        except Exception as e:
            logger.error(f"{Fore.RED}Error in dependency_aware_rag: {e}{Style.RESET_ALL}")
            return {
                "can_answer": False,  # 安全起见，出错时不轻易停止
                "current_understanding": f"Error during analysis: {str(e)}",
            }


    def generate_answer(self, question: str, history: List[Dict]) -> str:
        """Generate final answer based on the reasoning chain."""
        history_text = self._format_history_for_llm(history)

        debug_message = history_text
        print(debug_message)  ####################################  debug

        try:
            prompt = f"""
            You are a strict answer generator. You must generate the final answer based on the provided reasoning process.
            
            Question: {question}

            Reasoning Process:
            {history_text}

            【Strict Constraints】:
            1. Give ONLY the direct answer. DO NOT explain or provide any additional context.
            2. If the answer is a name, date, or number, output JUST that entity.
            3. If the answer is a simple yes/no, just say "Yes" or "No".
            4. If the answer requires a brief phrase, make it as concise as possible.
                       
            Concise Answer: """

            print(f'''  - Final Answer:{get_response_with_retry(prompt)}''')
            return get_response_with_retry(prompt)
        except Exception as e:
            logger.error(f"{Fore.RED}Error generating answer: {e}{Style.RESET_ALL}")
            return ""

    def _sort_dependencies(self, dependencies: List[str], query) -> List[Tuple]:
        """
        given a list of dependencies and the original query,
        sort the dependencies in a topological order, that is solving a dependency A relies on the solution of the dependent dependency B,
        then B should be before A in the sorted string.

        Args:
            dependencies: List[str]
            query: str

            
        For example, if the question is "What is the mayor of the capital of France?",
        the input dependencies for this question are:
        - The capital of France
        - The mayor of this capital

        Then the output should be:
        - The capital of France
        - The mayor of this capital

        there are two steps to solve this problem:
        1. generate the dependency pairs that dependency A relies on dependency B
        2. use graph-based algorithm to sort the dependencies in a topological order

        For example, answering the question "What is the mayor of the capital of France?"
        the input dependencies are:
        - The capital of France
        - The mayor of this capital

        Then the dependency pairs are:
        - [(1, 0)]
        because the mayor of the capital of France relies on the capital of France

        Then the topological order is computed by the self._topological_sort function, which is a graph-based algorithm. The output is a list of indices of the dependencies in the topological order.
        In this case, the output is:
        [0, 1]

        The sorted dependencies are thus:
        - The capital of France
        - The mayor of this capital
        """

        # Step 1: generate the dependency pairs by prompting LLMs
        prompt = f"""
        Given the question:
        Question: {query}

        and its decomposed dependencies:
        Dependencies: {dependencies}

        Please output the dependency pairs that dependency A relies on dependency B, if any. If no dependency pairs are found, output an empty list.

        format your response as a JSON object with these keys:
        - "dependency_pairs": list of tuples of integers (e.g., [[0, 1]])
        """
        response = get_response_with_retry(prompt)
        result = fix_json_response(response)
        dependency_pairs = result["dependency_pairs"]

        # Step 2: use graph-based algorithm to sort the dependencies in a topological order
        sorted_dependencies = self._topological_sort(dependencies, dependency_pairs)
        return sorted_dependencies

    @staticmethod
    def _topological_sort(dependencies: List[str], dependencies_pairs: List[Tuple[int, int]]) -> List[str]:
        """
        Use graph-based algorithm to sort the dependencies in a topological order.
        Args:
            dependencies: List[str]
            dependencies_pairs: List[Tuple[int, int]]
        Returns:
            List[str]
        """
        graph = {dep: [] for dep in dependencies}

        for dependent_idx, dependency_idx in dependencies_pairs:
            if dependent_idx < len(dependencies) and dependency_idx < len(dependencies):
                dependent = dependencies[dependent_idx]
                dependency = dependencies[dependency_idx]
                graph[dependency].append(dependent)  # dependency -> dependent

        visited = set()
        stack = []

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in graph[node]:
                dfs(neighbor)
            stack.append(node)

        for node in graph:
            if node not in visited:
                dfs(node)

        return stack[::-1]

    def _retrieve_with_filter(self, query: str, retrieved_chunks_set: set) -> list:
        """
        Retrieve top_k unique chunks not in retrieved_chunks_set. If not enough unique chunks, return as many as possible.
        """
        all_results = self.retrieve(query)
        unique_results = []
        idx = self.top_k
        # If not enough unique in top_k, keep expanding
        while len(unique_results) < self.top_k and idx <= len(self.corpus):
            # Expand retrieval window
            all_results = self.retrieve(query) if idx == self.top_k else self._retrieve_top_n(query, idx)
            unique_results = [chunk for chunk in all_results if chunk not in retrieved_chunks_set]
            idx += self.top_k
        return unique_results[:self.top_k]

    def _retrieve_top_n(self, query: str, n: int) -> list:
        """Retrieve top-n results for a query (helper for filtering)."""
        # Temporarily override top_k
        old_top_k = self.top_k
        self.top_k = n
        results = self.retrieve(query)
        self.top_k = old_top_k
        return results

    def answer_question(self, question: str) -> Tuple[str, List[str], int]:

        # --- 初始化变量 ---
        history = []  # [New] 用于存储结构化的推理链
        dependency_analysis_history = []
        last_contexts = []
        retrieval_history = []
        round_count = 0
        retrieved_chunks_set = set() if self.filter_repeats else None  # Track retrieved chunks if filtering

        print(f"\n\n{Fore.CYAN}{self.MODEL_NAME} answering: {question}{Style.RESET_ALL}\n\n")

        # ===============================================
        # == Stage 1: Warm up retrieval (作为第0步或初始背景) ==
        # ===============================================
        if self.filter_repeats:
            new_contexts = self._retrieve_with_filter(question, retrieved_chunks_set)
            for chunk in new_contexts:
                retrieved_chunks_set.add(chunk)
        else:
            new_contexts = self.retrieve(question)

        last_contexts = new_contexts

        # [New Logic] 处理 Warm-up 步骤
        # 即使是 Warm-up，我们也可以把它看作是对原始问题的一次直接尝试
        warmup_step_result = self.process_step(question, question, new_contexts)

        # 将 Warm-up 结果加入历史
        history.append({
            "step_type": "initial_attempt",
            "query": question,
            "summary": warmup_step_result["summary"],
            "answer": warmup_step_result["answer"]
        })
        # [Modified] 直接传入 history，不需要再手动生成 info_summary 字符串了
        analysis = self.warm_up_analysis(question, history)

        if analysis["can_answer"]:
            # In this case, the question can be answered with simple fact retrieval, without any dependency analysis
            print(
                f"Warm-up analysis indicate the question can be answered with simple fact retrieval, without any dependency analysis.")
            answer = self.generate_answer(question, history)  # 传入 history
            self.last_dependency_analysis = []
            self.last_history = history
            return answer, last_contexts, round_count
        else:
            logger.info(
                f"Warm-up analysis indicate the requirement of deeper reasoning-enhanced RAG. Now perform analysis with logical dependency graph.")
            logger.info(f"Dependencies: {', '.join(analysis.get('dependencies', []))}")

            # sort the dependencies, by first constructing the dependency graphs, then use topological sort to get the sorted dependencies
            sorted_dependencies = self._sort_dependencies(analysis["dependencies"], question)
            dependency_analysis_history.append({"sorted_dependencies": sorted_dependencies})
            logger.info(f"Sorted dependencies: {sorted_dependencies}\n\n")
        # ===============================================
        # == Stage 2: agentic iterative retrieval ==
        idx = 0  # used to track the current dependency index

        while round_count < self.max_rounds and idx < len(sorted_dependencies):
            round_count += 1

            current_query = sorted_dependencies[idx]
            if self.filter_repeats:
                new_contexts = self._retrieve_with_filter(current_query, retrieved_chunks_set)
                for chunk in new_contexts:
                    retrieved_chunks_set.add(chunk)
            else:
                new_contexts = self.retrieve(current_query)
            last_contexts = new_contexts  # Save current contexts

            # [New Logic] 处理当前步骤
            step_result = self.process_step(question, current_query, new_contexts)

            # Generate or refine information summary with new contexts
            # 保存到数组/栈
            history.append({
                "query": current_query,
                "step_type": "reasoning_step",
                "summary": step_result["summary"],
                "answer": step_result["answer"]
            })

            logger.info(f"Agentic retrieval at round {round_count} - Sub-answer: {step_result['answer']}")

            # [New Logic] 判别器使用 history
            analysis = self.dependency_aware_rag(question, history, sorted_dependencies, idx)

            retrieval_history.append({
                "round": round_count,
                "query": current_query,
                "contexts": new_contexts,
            })

            dependency_analysis_history.append({
                "round": round_count,
                "query": current_query,
                "analysis": analysis
            })

            if analysis["can_answer"]:
                # Generate and return final answer
                answer = self.generate_answer(question, history)  # 传入 history
                self.last_dependency_analysis = []  # 需要根据你的需求适配 log
                self.last_history = history
                return answer, last_contexts, round_count
            else:
                idx += 1

        # If max rounds reached, generate best possible answer
        logger.info(f"Reached maximum rounds ({self.max_rounds}). Generating final answer...")
        answer = self.generate_answer(question, history)
        self.last_history = history
        return answer, last_contexts, round_count

    def _format_history_for_llm(self, history: List[Dict[str, Any]]) -> str:
        """
        将推理历史列表格式化为清晰的字符串，供LLM阅读。

        Args:
            history: 包含每一步推理信息的列表，格式如下：
                     [{'query': '...', 'summary': '...', 'answer': '...'}, ...]
        Returns:
            String representation of the reasoning chain.
        """
        formatted_text = ""
        for i, step in enumerate(history):
            formatted_text += f"Step {i + 1}:\n"
            formatted_text += f"  - Sub-Query: {step['query']}\n"
            formatted_text += f"  - Context Summary: {step['summary']}\n"
            formatted_text += f"  - Direct Answer: {step['answer']}\n\n"

        return formatted_text.strip()
