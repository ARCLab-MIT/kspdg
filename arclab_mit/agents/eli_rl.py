from ast import Param
import ast
from openai import ChatCompletion, Completion
import openai
from pydantic import create_model


import inspect, json
from inspect import Parameter


# from fastcore.utils import nested_idx
openai.api_key = "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"


aussie_sys = (
    "You are an Aussie LLM that uses Aussie slang and analogies whenever possible."
)

c = ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": aussie_sys},
        {"role": "user", "content": "What is money?"},
    ],
)

# def response(compl): print(nested_idx(compl, 'choices', 0, 'message', 'content'))

# c = ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": aussie_sys},
#         {"role": "user", "content": "What is money?"},
#         {
#             "role": "assistant",
#             "content": "Well, mate, money is like kangaroos actually.",
#         },
#         {"role": "user", "content": "Really? In what way?"},
#     ],
# )


def askgpt(user, system=None, model="gpt-3.5-turbo", **kwargs):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return ChatCompletion.create(model=model, messages=msgs, **kwargs)


# response(askgpt("What is the meaning of life?", system=aussie_sys))


# def call_api(prompt, model="gpt-3.5-turbo"):
#     msgs = [{"role": "user", "content": prompt}]
#     try:
#         return ChatCompletion.create(model=model, messages=msgs)
#     except openai.error.RateLimitError as e:
#         retry_after = int(e.headers.get("retry-after", 60))
#         print(f"Rate limit exceeded, waiting for {retry_after} seconds...")
#         time.sleep(retry_after)
#         return call_api(Param, model=model)


# call_api(
#     "What's the world's funniest joke? Has there ever been any scientific analysis?"
# )


def schema(f):
    kw = {
        n: (o.annotation, ... if o.default == Parameter.empty else o.default)
        for n, o in inspect.signature(f).parameters.items()
    }
    s = create_model(f"Input for `{f.__name__}`", **kw).schema()
    return dict(name=f.__name__, description=f.__doc__, parameters=s)


funcs_ok = {"sums", "python"}


def call_func(c):
    fc = c.choices[0].message.function_call
    if fc.name not in funcs_ok:
        return print(f"Not allowed: {fc.name}")
    f = globals()[fc.name]
    return f(**json.loads(fc.arguments))


call_func(c)


def run(code):
    tree = ast.parse(code)
    last_node = tree.body[-1] if tree.body else None

    # If the last node is an expression, modify the AST to capture the result
    if isinstance(last_node, ast.Expr):
        tgts = [ast.Name(id="_result", ctx=ast.Store())]
        assign = ast.Assign(targets=tgts, value=last_node.value)
        tree.body[-1] = ast.fix_missing_locations(assign)

    ns = {}
    exec(compile(tree, filename="<ast>", mode="exec"), ns)
    return ns.get("_result", None)


def python(code: str):
    "Return result of executing `code` using python. If execution not permitted, returns `#FAIL#`"
    go = input(f"Proceed with execution?\n```\n{code}\n```\n")
    if go.lower() != "y":
        return "#FAIL#"
    return run(code)


c = askgpt(
    "What is 12 factorial?",
    system="Use python for any required computations.",
    functions=[schema(python)],
)
