from openai import ChatCompletion,Completion
from fastcore.utils import nested_idx
from pydantic import create_model
import inspect, json
from inspect import Parameter
import ast

#######################################
# The OpenAI API
#######################################

def response(compl):
    """Returns the response from a completion"""
    print(nested_idx(compl, 'choices', 0, 'message', 'content'))


def askgpt(user, system=None, model="gpt-3.5-turbo", **kwargs):
    """Ask a single question to an OpenAI model and return the response"""
    msgs = []
    if system: msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return ChatCompletion.create(model=model, messages=msgs, **kwargs)


#######################################
# code interpreter
#######################################

def schema(f):
    """Describe a function as a JSON schema to be passed as a function
    call to openai models"""
    kw = {n:(o.annotation, ... if o.default==Parameter.empty else o.default)
          for n,o in inspect.signature(f).parameters.items()}
    s = create_model(f'Input for `{f.__name__}`', **kw).schema()
    return dict(name=f.__name__, description=f.__doc__, parameters=s)


def call_func(c, funcs_ok=None):
    """
        Call a function from a completion object.
        If `funcs_ok` is not None, only allow the functions in the list.
    """
    fc = c.choices[0].message.function_call
    if funcs_ok is not None and fc.name not in funcs_ok: 
        return print(f'Not allowed: {fc.name}')
    f = globals()[fc.name]
    return f(**json.loads(fc.arguments))


def run(code):
    """
    Execute the given code and return the result.

    Args:
        code (str): The code to execute.

    Returns:
        The result of executing the code, or None if there is no result.
    """
    tree = ast.parse(code)
    last_node = tree.body[-1] if tree.body else None
    
    # If the last node is an expression, modify the AST to capture the result
    if isinstance(last_node, ast.Expr):
        tgts = [ast.Name(id='_result', ctx=ast.Store())]
        assign = ast.Assign(targets=tgts, value=last_node.value)
        tree.body[-1] = ast.fix_missing_locations(assign)

    ns = {}
    exec(compile(tree, filename='<ast>', mode='exec'), ns)
    return ns.get('_result', None)


def python(code:str):
    "Return result of executing `code` using python. If execution not permitted, returns `#FAIL#`"
    go = input(f'Proceed with execution?\n```\n{code}\n```\n')
    if go.lower()!='y': return '#FAIL#'
    return run(code)


####################################
# PyTorch and Huggingface
####################################

def gen(tokr, model, p, maxlen=15, sample=True):
    """
    Generate text from a prompt using a Huggingface model.

    Args:
        tokr (Tokenizer): The tokenizer to use.
        model (PreTrainedModel): The model to use.
        p (str): The prompt to use.
        maxlen (int): The maximum number of tokens to generate.
        sample (bool): Whether to sample or not.

    Returns:
        The generated text.

    Example:
        tokr = AutoTokenizer.from_pretrained(mn)
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", 
                                                    device_map=0, 
                                                    load_in_8bit=True)
        p = "En un lugar de la mancha, de"
        gen(tokr, model, p, maxlen=15, sample=True)
    """
    toks = tokr(p, return_tensors="pt")
    res = model.generate(**toks.to("cuda"), max_new_tokens=maxlen, do_sample=sample).to('cpu')
    return tokr.batch_decode(res)
