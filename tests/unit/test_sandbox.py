"""Tests for the sandboxed code executor."""

from backend.core.sandbox import ExecutionResult, execute_code


class TestTimeout:

    def test_timeout_kills_slow_code(self, mock_titanic_df):
        code = "while True: pass"
        result = execute_code(code, mock_titanic_df, timeout=1)
        assert not result.success
        assert "timeout" in result.error.lower()
        assert not result.retryable

    def test_fast_code_succeeds(self, mock_titanic_df):
        result = execute_code("result = 42", mock_titanic_df, timeout=5)
        assert result.success


class TestOutputDetection:

    def test_scalar(self, mock_titanic_df):
        result = execute_code("result = df['Age'].mean()", mock_titanic_df)
        assert result.success and result.output_type == "scalar"

    def test_dataframe(self, mock_titanic_df):
        result = execute_code("result = df.head(3)", mock_titanic_df)
        assert result.success and result.output_type == "dataframe"

    def test_series(self, mock_titanic_df):
        result = execute_code("result = df['Age']", mock_titanic_df)
        assert result.success and result.output_type == "series"

    def test_string(self, mock_titanic_df):
        result = execute_code("result = 'hello world'", mock_titanic_df)
        assert result.success and result.output_type == "string"
        assert result.output == "hello world"

    def test_no_result_var(self, mock_titanic_df):
        result = execute_code("x = 42", mock_titanic_df)
        assert result.success and result.output_type == "none"


class TestErrorClassification:

    def test_key_error_is_retryable(self, mock_titanic_df):
        result = execute_code("result = df['NonExistentColumn'].mean()", mock_titanic_df)
        assert not result.success and result.retryable

    def test_type_error_is_retryable(self, mock_titanic_df):
        result = execute_code("result = df['Age'] + 'string'", mock_titanic_df)
        assert not result.success and result.retryable

    def test_syntax_error_is_not_retryable(self, mock_titanic_df):
        result = execute_code("def foo(", mock_titanic_df)
        assert not result.success and not result.retryable


class TestExecutionResult:

    def test_dataclass_fields(self):
        r = ExecutionResult(success=True, output=42, output_type="scalar", execution_time_ms=10)
        assert r.success and r.output == 42 and r.output_type == "scalar"


class TestDfIsolation:

    def test_original_df_not_mutated(self, mock_titanic_df):
        original_len = len(mock_titanic_df)
        execute_code("df.drop(df.index, inplace=True); result = len(df)", mock_titanic_df)
        assert len(mock_titanic_df) == original_len


class TestSandboxSecurity:

    def test_escape_via_builtins_dict_raises_keyerror(self, mock_titanic_df):
        # We manually bypass AST validator to test the sandbox isolation layer directly
        code = "result = __builtins__['open']('/etc/passwd').read()"
        res = execute_code(code, mock_titanic_df)
        assert not res.success
        assert "KeyError: 'open'" in res.error

    def test_escape_via_dynamic_import_raises_importerror(self, mock_titanic_df):
        # Test the _safe_import wrapper
        code = "result = __import__('os').system('echo hacked')"
        res = execute_code(code, mock_titanic_df)
        assert not res.success
        assert "ImportError" in res.error
        assert "blocked by sandbox" in res.error
