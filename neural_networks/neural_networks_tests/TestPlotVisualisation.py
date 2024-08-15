import unittest
import torch
from neural_networks.plot_visualisation import mean_distance, compute_pourcentage_error

class TestPlotVisualisation(unittest.TestCase):
    
    # mean_distance
    def test_zero_distance(self):
        """Test case where predictions and targets are the same."""
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])
        self.assertEqual(mean_distance(predictions, targets), 0.0)

    def test_positive_distance(self):
        """Test case with known positive distances."""
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([2.0, 3.0, 4.0])
        expected_distance = 1.0  # Mean absolute distance = 1
        self.assertEqual(mean_distance(predictions, targets), expected_distance)

    def test_negative_distance(self):
        """Test case with predictions having negative values."""
        predictions = torch.tensor([-1.0, -2.0, -3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])
        expected_distance = 4.0  # Mean absolute distance = 4
        self.assertEqual(mean_distance(predictions, targets), expected_distance)

    def test_mixed_values(self):
        """Test case with mixed positive and negative values."""
        predictions = torch.tensor([1.0, -2.0, 3.0])
        targets = torch.tensor([-1.0, 2.0, -3.0])
        expected_distance = 4.0  # Mean absolute distance = 4
        self.assertEqual(mean_distance(predictions, targets), expected_distance)

    def test_large_values(self):
        """Test case with large values."""
        predictions = torch.tensor([1e6, 2e6, 3e6])
        targets = torch.tensor([2e6, 3e6, 4e6])
        expected_distance = 1e6  # Mean absolute distance = 1e6
        self.assertEqual(mean_distance(predictions, targets), expected_distance)
    
    # compute_pourcentage_error

    def test_zero_error(self):
        """Test case where predictions and targets are the same."""
        predictions = torch.tensor([10.0, 20.0, 30.0])
        targets = torch.tensor([10.0, 20.0, 30.0])
        expected_error = 0.0
        error, abs_error = compute_pourcentage_error(predictions, targets)
        self.assertEqual(error, expected_error)
        self.assertEqual(abs_error, expected_error)

    def test_positive_error(self):
        """Test case with known positive error."""
        predictions = torch.tensor([11.0, 22.0, 33.0])
        targets = torch.tensor([10.0, 20.0, 30.0])
        expected_error = (0.1 + 0.1 + 0.1) / 3 * 100  # Mean relative error = 10%
        error, abs_error = compute_pourcentage_error(predictions, targets)
        self.assertAlmostEqual(error, expected_error, places=5)
        self.assertAlmostEqual(abs_error, expected_error, places=5)

    def test_negative_error(self):
        """Test case with negative predictions."""
        predictions = torch.tensor([-11.0, -22.0, -33.0])
        targets = torch.tensor([-10.0, -20.0, -30.0])
        expected_error = (0.1 + 0.1 + 0.1) / 3 * 100  # Mean relative error = 10%
        error, abs_error = compute_pourcentage_error(predictions, targets)
        self.assertAlmostEqual(error, expected_error, places=5)
        self.assertAlmostEqual(abs_error, expected_error, places=5)

    def test_mixed_values(self):
        """Test case with mixed positive and negative values."""
        predictions = torch.tensor([11.0, -19.0, 31.0])
        targets = torch.tensor([10.0, -20.0, 30.0])
        expected_error = (0.1 + 0.05 + 0.0333) / 3 * 100  # Approximate mean relative error
        error, abs_error = compute_pourcentage_error(predictions, targets)
        self.assertAlmostEqual(error, expected_error, places=2)
        self.assertAlmostEqual(abs_error, expected_error, places=2)

    def test_large_values(self):
        """Test case with large values."""
        predictions = torch.tensor([1e6, 2e6, 3e6])
        targets = torch.tensor([1.1e6, 2.1e6, 3.1e6])
        expected_error = (1/11 + 1/21 + 1/31) / 3 * 100  # Mean relative error
        error, abs_error = compute_pourcentage_error(predictions, targets)
        self.assertAlmostEqual(error, expected_error, places=5)
        self.assertAlmostEqual(abs_error, expected_error, places=5)

    def test_division_by_zero(self):
        """Test case to handle division by zero."""
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([0.0, 0.0, 0.0])
        with self.assertRaises(ZeroDivisionError):
            compute_pourcentage_error(predictions, targets)

if __name__ == "__main__":
    unittest.main()
