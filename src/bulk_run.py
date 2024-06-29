import os


def main() -> None:
    command_base = "python src/entrypoint.py"
    ks = [2, 3, 4, 5, 7, 8, 9, 10]
    for k in ks:
        command = command_base + f" --k {k}"
        print(command)
        os.system(command)


if __name__ == "__main__":
    main()
