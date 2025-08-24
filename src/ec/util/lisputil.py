from typing import List

class LispUtil:
    @staticmethod
    def splitArguments(argsString: str) -> List[str]:
        args = []
        head = 0

        while head < len(argsString):
            if argsString[head] == '(':
                unbalance = 1
                i = head + 1
                while i < len(argsString):
                    if argsString[i] == '(':
                        unbalance += 1
                    elif argsString[i] == ')':
                        unbalance -= 1
                        if unbalance == 0:
                            args.append(argsString[head:i+1])
                            head = i + 2
                            break
                    i += 1
                else:
                    # Handle unbalanced parentheses
                    args.append(argsString[head:])
                    break
            else:
                tail = argsString.find(' ', head)
                if tail == -1:
                    tail = len(argsString)
                args.append(argsString[head:tail])
                head = tail + 1

        return args