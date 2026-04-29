import pandas as pd
import pandas.testing as pdt
import unittest

from src.data_cleaner import DataCleaner


def make_sample_df() -> pd.DataFrame:
    """Create a small DataFrame for testing.

    The DataFrame intentionally contains missing values, extra whitespace
    in a text column, and an obvious numeric outlier.
    """
    return pd.DataFrame(
        {
            "name": [" Alice ", "Bob", None, " Carol  "],
            "age": [25, None, 35, 120],  # 120 is a likely outlier
            "city": ["SCL", "LPZ", "SCL", "LPZ"],
        }
    )


class TestDataCleaner(unittest.TestCase):
    """Test suite for DataCleaner class."""

    def test_example_trim_strings_with_pandas_testing(self):
        """Ejemplo de test usando pandas.testing para comparar DataFrames completos.
        
        Este test demuestra cómo usar pandas.testing.assert_frame_equal() para comparar
        DataFrames completos, lo cual es útil porque maneja correctamente los índices,
        tipos de datos y valores NaN de Pandas.
        """
        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  ", "Carol"],
            "age": [25, 30, 35]
        })
        cleaner = DataCleaner()
        
        result = cleaner.trim_strings(df, ["name"])
        
        # DataFrame esperado después de trim
        expected = pd.DataFrame({
            "name": ["Alice", "Bob", "Carol"],
            "age": [25, 30, 35]
        })
        
        # Usar pandas.testing.assert_frame_equal() para comparar DataFrames completos
        # Esto maneja correctamente índices, tipos y estructura de Pandas
        pdt.assert_frame_equal(result, expected)

    def test_example_drop_invalid_rows_with_pandas_testing(self):
        """Ejemplo de test usando pandas.testing para comparar Series.
        
        Este test demuestra cómo usar pandas.testing.assert_series_equal() para comparar
        Series completas, útil cuando queremos verificar que una columna completa tiene
        los valores esperados manteniendo los índices correctos.
        """
        df = pd.DataFrame({
            "name": ["Alice", None, "Bob"],
            "age": [25, 30, None],
            "city": ["SCL", "LPZ", "SCL"]
        })
        cleaner = DataCleaner()
        
        result = cleaner.drop_invalid_rows(df, ["name"])
        
        # Verificar que la columna 'name' ya no tiene valores faltantes
        # Los índices después de drop_invalid_rows son [0, 2] (se eliminó la fila 1)
        expected_name_series = pd.Series(["Alice", "Bob"], index=[0, 2], name="name")
        
        # Usar pandas.testing.assert_series_equal() para comparar Series completas
        # Esto verifica valores, índices y tipos correctamente
        pdt.assert_series_equal(result["name"], expected_name_series, check_names=True)

    def test_drop_invalid_rows_removes_rows_with_missing_values(self):
        df = make_sample_df()
        cleaner = DataCleaner()
        result = cleaner.drop_invalid_rows(df, ["name", "age"])
        self.assertEqual(result["name"].isna().sum(), 0)
        self.assertEqual(result["age"].isna().sum(), 0)
        self.assertLess(len(result), len(df))

    def test_drop_invalid_rows_raises_keyerror_for_unknown_column(self):
        df = make_sample_df()
        cleaner = DataCleaner()
        with self.assertRaises(KeyError):
             cleaner.drop_invalid_rows(df, ["does_not_exist"])

    def test_trim_strings_strips_whitespace_without_changing_other_columns(self):
        df = make_sample_df().dropna(subset=["name"])
        cleaner = DataCleaner()
        result = cleaner.trim_strings(df, ["name"])
        self.assertEqual(df.loc[0, "name"], " Alice ")
        self.assertEqual(df.loc[1, "name"], "Bob")
        self.assertEqual(df.loc[3, "name"], " Carol  ")
        self.assertEqual(result.loc[0, "name"], "Alice")
        self.assertEqual(result.loc[1, "name"], "Bob")
        self.assertEqual(result.loc[3, "name"], "Carol")
        pdt.assert_series_equal(result["city"], df["city"])

    def test_trim_strings_raises_typeerror_for_non_string_column(self):
        df = make_sample_df()
        cleaner = DataCleaner()
        with self.assertRaises(TypeError):
             cleaner.trim_strings(df, ["age"])

    def test_remove_outliers_iqr_removes_extreme_values(self):
        df = make_sample_df().dropna(subset=["age"])
        cleaner = DataCleaner()
        result = cleaner.remove_outliers_iqr(df, "age", factor=1.5)
        self.assertLessEqual(len(result), len(df))
        self.assertIn(25, result["age"].values)

    def test_remove_outliers_iqr_raises_keyerror_for_missing_column(self):
        df = make_sample_df()
        cleaner = DataCleaner()
        with self.assertRaises(KeyError):
             cleaner.remove_outliers_iqr(df, "salary")

    def test_remove_outliers_iqr_raises_typeerror_for_non_numeric_column(self):
        df = make_sample_df()
        cleaner = DataCleaner()
        with self.assertRaises(TypeError):
             cleaner.remove_outliers_iqr(df, "city")
if __name__ == "__main__":
    unittest.main()