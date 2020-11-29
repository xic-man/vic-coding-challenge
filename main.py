from functions import get_data, generate_poem
import random
import math

print("Victorian coding challenge: Wikipedia based data extraction and poem generation")

print("\nIf program is loading for a long time, try pressing any key\n")

while True:
    print("Following input must be either U (user provided name), r (required names), s (suggested names), or h (help and additional information) ")
    try:
        loop_user_input = input("\nWhat people should the script be run with?: ")
    except EOFError:  # Handling inputs such as ctrl C (^C)
        print("\nEnd of file condition typed, exiting...")
        exit()
    if loop_user_input == "" or loop_user_input.lower()[0] == "u":  # If the user wants to type the people to be run
        full_name_list = [None]*math.factorial(8)  # Extremely inefficient, generates a list with 40320 None's in it
        break
    elif loop_user_input.lower()[0] == "h":
        print("\nName input command list (case insensitive)")
        print("  ├────u to set user input, allowing the user to type a name to be parsed into the algorithm")
        print("  ├────r to run the script with all the names that are required under the competition rules")
        print("  └────s to run the script with an assortment of names intended to challenge and test the capabilities of the script\n")

    elif loop_user_input.lower()[0] == "r":  # If the user selects only required people to be run
        full_name_list = [
            "jacinda ardern",
            "albert einstein",
            "serena williams",
            "franklin roosevelt",
            "mark viduka",
            "marie curie"
        ]
        break
    elif loop_user_input.lower()[0] == "s":  # If the user selects suggested people to be run
        full_name_list = [  # A list of people used to test the capabilities of the script
            "john lennon",
            "barack obama",
            "isaac newton",
            "john snow",
            "paul klee",
            "daniel j. boorstin",
            "neal gabler",
            "david letterman",
            "lebron james",
            "paris hilton",
            "srinivasa ramanujan",
            "guido van rossum",
            "bjarne stroustrup",
            "neil degrasse tyson"
            #  "André 3000"
        ]
        break
    else:
        print(f"Unsupported command typed ({loop_user_input}), please try again")
random.shuffle(full_name_list)  # Randomising order of list

# A list of settings to be parsed through the generate_poem function, with default values specified
poem_settings = {
    "Poem order": "normal",  # Can be set to either normal or random
    "AI fill": True,  # Can be True or False
    "Set poem order": "aaaabbcc",  # User set poem rhyming order
    # Rhyming settings
    "Number of words": 10,  # number of words to return
    "Words to generate": 10,  # if this is larger than no of words, then the output is slightly randomised
    "Number of syllables": 2  # Default filter is >, so upwards of the specified value
    }

# User Settings loop
while True:
    try:
        poem_settings_input = input("Poem settings (h for commands, e/enter to exit): ")
    except EOFError:  # Handling inputs such as ctrl C (^C)
        print("\nEnd of file condition typed, exiting...")
        exit()
    if poem_settings_input == "" or poem_settings_input.lower()[0] == "e":
        break
    elif poem_settings_input.lower()[0] == "h":  # Help settings
        print("\nPoem Command list (case insensitive)")
        print("├h for list of commands")
        print("├o to set rhyming scheme of poem")
        print("│└──── Rhyming scheme must have 8 characters, with 4 a's, 2 b's, and 2 c's (e.g aaaabbcc). Case insensitive.")
        print("├r to randomize order of poem lines")
        print("├a to turn off AI poem generation")
        print("├s to change poem syllable rhyming settings")
        print("│├────n to change number of words to generate (not more words in the poem, will")
        print("││     increase pool of words which helps with randomness, may decrease quality of rhymes)")
        print("│└────s to change number of syllables of all rhyming words")
        print("└e/enter to exit\n")

    elif poem_settings_input.lower() == "r":  # Toggle random poem order
        if poem_settings["Poem order"] == "random":
            poem_settings["Poem order"] = "normal"
            print("Order of poem set to normal")
        else:
            poem_settings["Poem order"] = "random"
            print("Order of poem randomized")

    elif poem_settings_input.lower() == "a":  # Toggle AI poem fill
        if not poem_settings["AI fill"]:
            poem_settings["AI fill"] = True
            print("AI poem fill set to True")
        else:
            poem_settings["AI fill"] = False
            print("AI poem fill set to False")

    elif poem_settings_input.lower()[0] == "o":  # User set poem rhyming scheme
        if poem_settings["Poem order"] == "random":  # Order cannot be random and user set at the same time
            poem_settings["Poem order"] = "normal"
            print("Warning: Cannot have random order and set order at the same time. Order is now set to normal...")
        print("\nRhyming scheme must have 8 characters, with 4 a's, 2 b's, and 2 c's (e.g aaaabbcc). Case insensitive.\n")
        while True:  # User set poem rhyming scheme loop
            try:
                rhy_scheme = input("Set rhyming scheme of poem (e to exit): ")
            except EOFError:  # Handling inputs such as ctrl C (^C)
                print("\nEnd of file condition typed, exiting...")
                exit()
            if rhy_scheme == "" or rhy_scheme.lower()[0] == "e":
                break
            # Checking number of characters is 8
            if len(rhy_scheme) == 8:
                # Checking that there are 4 a's, 2 b's, and 2 c's
                if rhy_scheme.lower().count("a") == 4 and rhy_scheme.lower().count("b") == 2 and rhy_scheme.lower().count("c") == 2:
                    poem_settings["Set poem order"] = rhy_scheme.lower()
                    print(f"Rhyming scheme submitted: {rhy_scheme.lower()}")
                    break
                else:
                    # Defining variables used for error message
                    num_a = rhy_scheme.lower().count("a")
                    num_b = rhy_scheme.lower().count("b")
                    num_c = rhy_scheme.lower().count("c")
                    print(f"Incorrect ratio of characters (Currently {num_a} a's, {num_b} b's, and {num_c} c's)")
            else:
                print(f"Check number of characters and try again (Currently {len(rhy_scheme)})")

    elif poem_settings_input.lower()[0] == "s":  # User set syllable settings
        while True:  # User set syllable settings loop
            try:
                syl_set_input = input("Change syllables settings of rhyming words (e/enter to exit, h for help): ")
            except EOFError:  # Handling inputs such as ctrl C (^C)
                print("\nEnd of file condition typed, exiting...")
                exit()
            if syl_set_input == "" or syl_set_input.lower()[0] == "e":
                break

            elif syl_set_input.lower()[0] == "n":  # User set number of words to generate
                while True:  # User set number of words to generate loop
                    print("\nFollowing user input must be either an integer (number), e, or enter\n")
                    try:
                        num_words = input("Please type number of words to generate: ")
                    except EOFError:  # Handling inputs such as ctrl C (^C)
                        print("\nEnd of file condition typed, exiting...")
                        exit()
                    if num_words == "" or num_words.lower()[0] == "e":
                        break
                    try:  # Making sure user input is an integer
                        poem_settings["Words to generate"] = int(num_words)
                        print(f"Number of words set to {int(num_words)}")
                        break
                    except ValueError:  # If input is a string or other
                        print("Please type in either an integer (number), e, or enter")

            elif syl_set_input.lower()[0] == "s":  # User set number of syllables
                while True:  # User set number of syllables loop
                    print("\nFollowing user input must be either an integer (number), e, or enter\n")
                    try:
                        num_syllables = input("Please type number of syllables: ")
                    except EOFError:  # Handling inputs such as ctrl C (^C)
                        print("\nEnd of file condition typed, exiting...")
                        exit()
                    try:  # Making sure user input is an integer
                        poem_settings["Number of syllables"] = int(num_syllables)
                        print(f"Number of syllables set to {int(num_syllables)}")
                        break
                    except ValueError:  # If input is a string or other
                        print("Please type in either an integer (number), e, or enter")
                    if num_syllables == "" or num_syllables.lower()[0] == "e":
                        break

            elif syl_set_input.lower()[0] == "h":   # Help/settings
                print("\nSyllable Command list (case insensitive)")
                print("  ├────n to change number of words to generate (not more words in the poem, will")
                print("  │    increase pool of words which helps with randomness, may decrease quality of rhymes)")
                print("  └────s to change number of syllables of all rhyming words\n")
            else:
                print(f"Unsupported command typed ({syl_set_input}), please check spelling and try again")

    else:
        print(f"Unsupported command typed ({poem_settings_input}), please check spelling and try again")

print("\nPoem settings to be used:")
for i in range(len(poem_settings)):  # iterates through the length of the dict and prints the dict title and its contained data
    print(list(poem_settings)[i] + ": " + str(list(poem_settings.values())[i]))
print()

for full_name in full_name_list:  # Main loop
    while full_name is None:  # For user input
        try:
            full_name = input("Please type a name: ")
        except EOFError:  # Handling inputs such as ctrl C (^C)
            print("\nEnd of file condition typed, exiting...")
            exit()
        print()
    all_data = get_data(full_name)  # Running get_data function to obtain all data
    if all_data is None:
        continue  # Jumping to the next loop if no data is returned
    print("Name:", all_data[6].title(), all_data[7].title())
    print("URL:", all_data[8])
    birth_date = all_data[0]
    birth_year = "year"
    birth_month = None
    birth_day = None
    if birth_date != "unknown":
        birth_year = str(birth_date).split("-")[0]
        birth_month = str(birth_date).split("-")[1]
        birth_day = str(birth_date).split("-")[2]
        print("Birthday:", f"{birth_day}/{birth_month}/{birth_year}")
    else:
        print("Birthday: Unknown")
    age = all_data[1]
    alive_or_dead = all_data[2]
    career = all_data[3]
    gender = all_data[4]
    subscience = all_data[5]
    first_name = all_data[6]
    last_name = all_data[7]
    if alive_or_dead == "alive":
        print("Current age:", age, "(Currently alive)")
    elif alive_or_dead == "dead":
        print("Age at death:", age)
    elif alive_or_dead == "unknown":
        print(f"Age: Unknown (Age could not be obtained for {full_name})")
    if career == "unknown":
        print(f"Career: {career.title()} (Career could not be obtained for {full_name})")
    else:
        print("Career: ", career.title())
    if subscience is not None:
        print("Subscience: ", subscience.title())
    print(f"Gender: {gender.title()}\n")

    all_data = [full_name, first_name, last_name, age, birth_year, alive_or_dead, career, gender, subscience]
    generate_poem(all_data, poem_settings)  # Parsing all data into the poem generation function
    print("\n")
