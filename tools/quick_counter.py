import readchar

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'
BOLD = '\u001b[1m'

def count():
    count_required = 0
    count_broken = 0
    multiplier = 1

    while True:
        count_required = max(0, count_required)
        count_broken = max(0, count_broken)
        print(LINE_UP, end=LINE_CLEAR)
        print(f"{BOLD}Required: %d, Broken: %d" % (count_required, count_broken))
        key = readchar.readchar()

        if key == 'q':
            count_required = 0
            count_broken = 0
        if key == 'Q':
            break;
        if key == 'x':
            count_required += 1*multiplier
            multiplier = 1
        if key == 'c':
            count_broken += 1*multiplier
            multiplier = 1
        if key == 'C':
            count_broken -= 1*multiplier
            multiplier = 1
        if key == 'X':
            count_required -= 1*multiplier
            multiplier = 1
        if key == '1':
            multiplier = 1
        if key == '2':
            multiplier = 2
        if key == '3':
            multiplier = 3
        if key == '4':
            multiplier = 4
        if key == '5':
            multiplier = 5
        if key == '6':
            multiplier = 6
        if key == '7':
            multiplier = 7
        if key == '8':
            multiplier = 8
        if key == '9':
            multiplier = 9
        if key == '0':
            multiplier = 10


def main():
    count()


if __name__ == "__main__":
    main()
