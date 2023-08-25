# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import utils


def main():
    total, correct = utils.evaluate_places("birth_dev.tsv", ["London"] * 500)
    print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))


if __name__ == '__main__':
    main()
