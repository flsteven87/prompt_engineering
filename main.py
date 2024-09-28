import json
import os
from typing import List, Dict
from dotenv import load_dotenv
import openai
import dspy
from dspy.teleprompt import MIPROv2
from dspy.evaluate import answer_exact_match
from pydantic import BaseModel, Field
import logging
import time

logging.getLogger("httpx").setLevel(logging.WARNING)

class QuestionAnswer(BaseModel):
    question: str = Field(..., description="問題")
    answer: str = Field(..., description="答案")

class SyntheticData(BaseModel):
    data: List[QuestionAnswer] = Field(..., description="合成數據列表")

class SyntheticDataOutput(BaseModel):
    data: List[QuestionAnswer] = Field(..., description="合成數據列表")

class BrandExtractor:
    """
    用於使用 AI 模型和提示優化從文本中提取品牌名稱的類。
    """

    def __init__(self):
        """
        初始化 BrandExtractor 並設置必要的配置和模型。
        """
        load_dotenv()
        openai.api_key = os.environ['OPENAI_API_KEY']

        self.synthetic_model = "gpt-4o-2024-08-06"
        self.generation_model = "gpt-4o-2024-08-06"
        self.prediction_model = "gpt-4o-mini"

        self.task_description = "從文本中提取品牌。"
        self.categories = [
            "Nike", "Adidas", "Puma", "Asics", "Converse", "Reebok", 
            "Skechers", "Rtfkt", "Amazon", "Finishline", "Jordan", "Vans", 
            "Air Jordan", "Apple", "New Balance", "Netflix", "Yeezy", "Crocs", 
            "Manchester United", "Starbucks"
        ]
        self.questions_num = len(self.categories) * 5
        self.max_retries = 3
        self.retry_delay = 5

    def generate_synthetic_data(self) -> List[QuestionAnswer]:
        """
        使用指定 AI 模型生成合成訓練數據。
        
        返回：
            List[QuestionAnswer]：包含問題和答案的 QuestionAnswer 對象列表。
        """
        with open('prompt/synthetic_prompt.txt', 'r', encoding='utf-8') as file:
            synthetic_prompt = file.read()

        # 將必要的參數插入到 prompt 中
        synthetic_prompt = synthetic_prompt.format(
            task_description=self.task_description,
            categories=json.dumps(self.categories, ensure_ascii=False),
            questions_num=self.questions_num
        )

        messages = [
            {"role": "system", "content": "你是一個專門生成合成數據的AI助手。"},
            {"role": "user", "content": synthetic_prompt}
        ]

        for attempt in range(self.max_retries):
            try:
                client = openai.OpenAI()
                response = client.chat.completions.create(
                    model=self.synthetic_model,
                    temperature=0,
                    messages=messages,
                    tools=[
                        openai.pydantic_function_tool(
                            SyntheticDataOutput,
                            name="output_synthetic_data",
                            description="輸出生成的合成數據。"
                        )
                    ]
                )

                tool_call = response.choices[0].message.tool_calls[0]
                if tool_call.function.name == "output_synthetic_data":
                    try:
                        synthetic_data = json.loads(tool_call.function.arguments)['data']
                        logging.info(f"模型輸出 (狀態: {response.choices[0].finish_reason}): {synthetic_data[:100]}...")  # 只記錄前100個字符
                        return [QuestionAnswer(question=item['question'], answer=item['answer']) for item in synthetic_data]
                    except json.JSONDecodeError as json_error:
                        logging.error(f"JSON 解析錯誤 (嘗試 {attempt + 1}/{self.max_retries}): {str(json_error)}")
                        logging.error(f"原始回應: {tool_call.function.arguments[:500]}...")  # 記錄前500個字符
                else:
                    raise ValueError("未找到預期的函數調用")

            except Exception as e:
                logging.error(f"處理合成數據時發生錯誤 (嘗試 {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logging.error("達到最大重試次數，無法生成合成數據")
                    return []

        return []  # 如果所有嘗試都失敗，返回空列表

    def setup_dspy_models(self):
        """
        設置用於提示生成和任務執行的 DSPy 模型。
        """
        self.prompt_llm = dspy.OpenAI(model=self.generation_model, api_key=os.environ['OPENAI_API_KEY'])
        self.task_llm = dspy.OpenAI(model=self.prediction_model, api_key=os.environ['OPENAI_API_KEY'])
        dspy.settings.configure(lm=self.task_llm)

    def create_question_classification_model(self):
        """
        使用 DSPy 創建並返回 QuestionClassification 模型。
        
        返回：
            dspy.Module：QuestionClassification 模型。
        """
        class QuestionLabel(dspy.Signature):
            question = dspy.InputField(desc="要分類的輸入問題")
            answer = dspy.OutputField(desc="問題的分配類別或標籤")

        class QuestionClassification(dspy.Module):
            def __init__(self):
                super().__init__()
                self.classifier = dspy.Predict(QuestionLabel)

            def forward(self, question: str):
                return self.classifier(question=question)

        return QuestionClassification()

    def optimize_prompt(self, dataset: List[QuestionAnswer]):
        """
        使用 MIPROv2 優化提示並將最終使用的提示保存到文件。
        
        參數：
            dataset (List[QuestionAnswer])：用於訓練的合成數據集。
        """
        few_shot_examples = [dspy.Example({'question': q.question, 'answer': q.answer}) for q in dataset]
        trainset = [x.with_inputs('question') for x in few_shot_examples]

        teleprompter = MIPROv2(prompt_model=self.prompt_llm, task_model=self.task_llm, 
                               metric=answer_exact_match, num_candidates=10, 
                               init_temperature=1, verbose=True)

        compiled_program = teleprompter.compile(self.create_question_classification_model(), 
                                                trainset=trainset, requires_permission_to_run=False)

        # 使用優化後的模型進行測試
        test_question = "班長有什麼了不起，我小學也當過班長啊!"
        result = compiled_program(question=test_question)
        print(f"測試問題: {test_question}")
        print(f"模型回答: {result.answer}")

        # 保存優化後的模型
        compiled_program.save("compiled_program.json")
        print("優化後的模型已保存為 compiled_program.json")

        # # 獲取最後使用的 prompt
        # final_prompt = self.task_llm.inspect_history(1)

        # # 將最終使用的 prompt 寫入文件
        # with open('prompt/final_optimized_prompt.txt', 'w', encoding='utf-8') as file:
        #     file.write("最終優化後使用的 prompt：\n\n")
        #     file.write(str(final_prompt))
        # print("最終優化後使用的 prompt 已寫入 prompt/final_optimized_prompt.txt")

        return compiled_program

    def run(self):
        """
        執行整個品牌提取過程。
        """
        dataset = self.generate_synthetic_data()
        if dataset:
            self.setup_dspy_models()
            self.optimize_prompt(dataset)
        else:
            print("無法生成合成數據，程序終止。")

if __name__ == "__main__":
    extractor = BrandExtractor()
    extractor.run()