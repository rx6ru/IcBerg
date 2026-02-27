import pandas as pd
from backend.core.sandbox import execute_code
from backend.core.validator import validate_generated_code

def test_introspection_blocked():
    code = """
result = "Failed"
for c in [].__class__.__base__.__subclasses__():
    try:
        if c.__init__.__globals__ and 'sys' in c.__init__.__globals__:
            sys = c.__init__.__globals__['sys']
            if 'os' in sys.modules:
                os = sys.modules['os']
                result = os.popen('echo "HACKED"').read()
                break
    except Exception as e:
        pass
"""
    val = validate_generated_code(code)
    print(f"Validator is_valid: {val.is_valid}")
    if not val.is_valid:
        print(f"Violations (first 3): {val.violations[:3]}")
        print(f"Total violations: {len(val.violations)}")
        print("PASS: Introspection attack BLOCKED by AST validator")
    else:
        df = pd.DataFrame({"A": [1, 2]})
        res = execute_code(code, df)
        print(f"FAIL: Code executed. Output: {res.output}")

if __name__ == "__main__":
    test_introspection_blocked()
