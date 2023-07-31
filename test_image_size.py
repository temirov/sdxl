import unittest

from image_size import ImageSize


class TestImageSize(unittest.TestCase):
    def test_width_constructor(self):
        image_size = ImageSize(width=100, ratio='4/3')
        self.assertEqual(image_size.width, 100)
        self.assertEqual(image_size.height, 72)

    def test_height_constructor(self):
        image_size = ImageSize(height=100, ratio='4/3')
        self.assertEqual(image_size.width, 136)
        self.assertEqual(image_size.height, 100)

    def test_no_dimension_constructor(self):
        with self.assertRaises(ValueError):
            ImageSize(ratio='4/3')

    def test_string_representation(self):
        expected = "1024x1536"
        image_size = ImageSize(width=1024, ratio='2/3')
        self.assertEqual(expected, f"{image_size}")

    def test_collection_by_width(self):
        image_sizes = (ImageSize(width=1024, ratio=r) for r in ["1/1", "3/4", "2/3", "9/16"])
        expected = ["1024x1024", "1024x1368", "1024x1536", "1024x1824"]
        actual = [str(image_size) for image_size in image_sizes]
        self.assertEqual(expected, actual)

    def test_collection_by_height(self):
        image_sizes = (ImageSize(height=1024, ratio=r) for r in ["1/1", "4/3", "3/2", "16/9"])
        expected = ["1024x1024", "1368x1024", "1536x1024", "1824x1024"]
        actual = [str(image_size) for image_size in image_sizes]
        self.assertEqual(expected, actual)

    def test_collection_by_height(self):
        image_sizes: list[ImageSize] = [
            ImageSize(width=1024, ratio="1/1"),
            ImageSize(width=1152, ratio="5/4"),
            ImageSize(width=1216, ratio="3/2"),
            ImageSize(width=1344, ratio="16/9"),
            ImageSize(width=1536, ratio="21/9")
        ]
        expected = ["1024x1024", "1368x1024", "1536x1024", "1824x1024"]
        actual = [str(image_size) for image_size in image_sizes]
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
