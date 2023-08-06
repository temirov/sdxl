from fractions import Fraction


class ImageSize:
    def __init__(self, width=None, height=None, ratio='1/1'):
        self.ratio = ratio
        self.ratio_fraction = Fraction(self.ratio)
        self.width = width
        self.height = height
        self.__post_init()

    @staticmethod
    def __round_to_nearest_eight(value: int) -> int:
        return round(value / 8) * 8

    def __calculate_height(self):
        return self.width / self.ratio_fraction.numerator * self.ratio_fraction.denominator

    def __calculate_width(self):
        return self.height * self.ratio_fraction.numerator / self.ratio_fraction.denominator

    def __post_init(self):
        if self.width is None and self.height is None:
            raise ValueError('Either width or height must be provided.')
        elif self.width is not None and self.height is None:
            self.height = self.__round_to_nearest_eight(self.__calculate_height())
        elif self.height is not None and self.width is None:
            self.width = self.__round_to_nearest_eight(self.__calculate_width())

    def __repr__(self) -> str:
        return f"{self.width}x{self.height} {self.ratio}"
