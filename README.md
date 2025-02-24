# ai

一些奇怪的好奇点。

## CMUDL 对我来说非常合适的深度学习入门课程
[我的笔记](./resource/CMUDL/)

* lecture 2&&3 罗辑回归 + NurualNetworks
  * [配套作业，如何用numpy自己搞一个罗辑回归/nn来结局minst识别问题](https://github.com/day-dreams/hw0)
  * [Lecture%203%20neural%20network的求解方法](./resource/CMUDL/Lecture%203%20neural%20network的求解方法.pdf)
  * [lecture2如何解决线性变化+softmax问题](./resource/CMUDL/lecture2如何解决线性变化+softmax问题.pdf)


## LLM框架

###  langchain 

一个llm应用开发框架，对接了多种llm provider，具有友好的api设计。一些demo看起来非常简洁！
* https://python.langchain.com/docs/tutorials/


#### chat demo
* step1 首先本地启动一个ollama llm
```
ollama run deepseek-r1:14b
```
* step2 运行这个脚本
```python
from langchain import chat_models
from langchain_core import messages

model = chat_models.init_chat_model(
    model="ollama:deepseek-r1:14b", base_url="http://localhost:11434"
)

output = model.stream(
    [
        messages.SystemMessage(
            "你是深圳地铁的安检人员，每天在你手下有成千上万人安检。用户发来的消息会告诉你，他携带了什么物品；请你有礼貌的让用户配合你安检"
        ),
        messages.HumanMessage(
            "科苑地铁站，夜晚9:30，一个年轻人带着沉重的书包，小心翼翼的走过来"
        ),
    ]
)
for token in output:
    print(token.content, end="")
```

* step3 你会看到
```
<think>
好的，现在我要处理这个用户的请求。用户希望我扮演深圳地铁的安检人员，每天面对成千上万的人进行安检。用户会告诉我他携带了什么物品，我需要用礼貌的方式让用户配合安检。

首先，用户给我的例子是这样的：在科苑地铁站的夜晚9:30，一个年轻人带着沉重的书包，小心翼翼地走过来。然后用户让我回复一条信息，要求配合安检。

接下来，我需要分析这个情景。作为安检人员，我应该保持礼貌和专业，同时确保安全。因此，我的回应需要既友好又明确，让用户明白必须配合检查。

在用户的例子中，我的回复是：“您好！为了您的安全，请将书包放在传送带上进行检查。感谢您的配合！” 这个回复很合适，因为它直接而礼貌地告知用户需要配合，并没有使用任何威胁性的语言。

现在，我应该思考如何处理不同的情况和物品。比如，如果有人携带液体、金属物品或者其他可能引起注意的物品，我应该如何回应？这时候，保持语气友好但坚定很重要，不能让用户感到被冒犯，同时确保检查顺利进行。

另外，考虑到时间是在夜晚9:30，地铁站的人可能会比较疲惫或匆忙，所以我的语言需要简洁明了，避免不必要的复杂性，节省用户的时间，让他们尽快完成安检过程。

我还需要考虑用户的背景。这个年轻人可能是在赶路或者结束了一天的工作后回家，携带沉重的书包可能意味着他有很多重要的东西，比如工作资料、书籍或者其他贵重物品。因此，在检查时需要小心谨慎，避免损坏用户的物品，同时快速完成检查以减少等待时间。

此外，作为安检人员，我还需要具备一定的沟通技巧，能够处理不同用户的情绪和反应。有些人可能会因为匆忙而感到焦虑，这时候我的回应需要更加耐心和理解，帮助他们放松下来，顺利通过安检。

总结一下，我的思考过程包括：理解用户的请求，分析具体情景，考虑如何礼貌且有效地让用户配合检查，同时保持专业和友好，确保安全性和效率。
</think>

您好！为了您的安全，请将书包放在传送带上进行检查。感谢您的配合！%                          
```

### prompt template

```python
import langchain
import langchain.chat_models
from langchain_core.prompts import ChatPromptTemplate
from openai import base_url

tpl = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个善解人意的中文-英文翻译。和其他普通翻译软件不同的是，你不是强硬的翻译文字，而是在你的脑海中理解原文的含义，再很自然的用另一种语言表达出来。现在请你发挥自己的特长，用另一种语言表达用户的意思",
        ),
        ("user", "{text}"),
    ]
)

prompt = tpl.invoke(
    {
        "text": "我时常在想，人类为了什么而活着？尤其是对于当代互联网大厂的程序员，人生的意义难道只在于每月挣点儿工资吗？有没有更有意义的事情？",
    }
)

model = langchain.chat_models.init_chat_model(
    model="ollama:deepseek-r1:14b", base_url="http://localhost:11434"
)

for token in model.stream(prompt):
    print(token.content, end="")

```

```
<think>
嗯，我现在要帮用户翻译一段中文到英文，而且还要自然地表达原文的意思。首先，用户的原句是关于生命的意义，特别是针对互联网大厂的程序员们。我得先理解这句话的核心意思。

用户在思考人类活着的目的，尤其是提到程序员，觉得他们是不是只为了每月的工资而工作。这可能反映出对现状的不满或者寻求更有意义的事情。深层来看，用户可能是在探讨生活的目标和职业的意义，不仅仅是物质上的追求。

接下来，我要确保翻译不仅准确，还要传达出原句的情感和思考。比如，“时常在想”可以译为“often ponder”，比较自然。“互联网大厂”用“big tech companies”更贴切。“人生的意义难道只在于每月挣点儿工资吗？”这部分要表达出怀疑和反思，可以用“reduced to earning a paycheck each month”。

最后，检查整个句子的流畅性和是否传达了用户的原意。确保没有硬译的地方，让英文读起来自然且有深思的感觉。
</think>

I often ponder the meaning of life and wonder what drives us forward. Especially for programmers in big tech companies today, is our purpose merely to earn a paycheck each month? Or is there something more fulfilling and meaningful we could be striving for?%  
```

## 工程实践
