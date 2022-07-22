from main import chat, course_recommendation
from main import peer_grouping
from main import class_type
import discord
import random


client = discord.Client()

TOKEN = "TOKEN"

class Question:
     def __init__(self, prompt, answer):
          self.prompt = prompt
          self.answer = answer

question_prompts = [
     "1. Can I park here?\n(a) Sorry, I did that.\n(b) It's the same place.\n(c) Only for half an hour.",
     "2. What colour will you paint the children's bedroom?\n(a) I hope it was right.\n(b) We can't decide.\n(c) It wasn't very difficult.",
     "3. I can't understand this email.\n (a) Would you like some help?\n (b) Don't you know?\n (c) I suppose you can.",
     "4. I'd like two tickets for tomorrow night.\n (a) How much did you pay?\n (b) Afternoon and evening.\n (c) I'll just (just in case meaning) check for you.",
     "5. Shall we go to the gym now?\n (a) I'm too tired.\n (b) It's very good.\n (c) Not at all.",
     "6. His eyes were ...... bad that he couldn't read the number plate of the car in front.\n (a) such\n (b) too\n (c) so\n (d) very",
     "7. The company needs to decide ...... and for all what its position is on this point.\n (a) here\n (b) once\n (c) first\n (d) finally",
     "8. Don't put your cup on the ...... of the table – someone will knock it off.\n (a) outside\n (b) edge\n (c) boundary\n (d) border",
     "9. I'm sorry - I didn't ...... to disturb you.\n (a) hope\n (b) think\n (c) mean\n (d) suppose",
     "10. The singer ended the concert ...... her most popular song.\n (a) by\n (b) with\n (c) in\n (d) as",
     "11. Would you mind ...... these plates a wipe before putting them in the cupboard?\n (a) making\n (b) doing\n (c) getting\n (d) giving",
     "12. I was looking forward ...... at the new restaurant, but it was closed.\n (a) to eat\n (b) to have eaten\n (c) to eating\n (d) eating",
     "13. ...... tired Melissa is when she gets home (home of idioms) from work, she always makes time (killing time synonym) to say goodnight to the children.\n (a) Whatever\n (b) No matter how\n (c) However much\n (d) Although",
     "14. It was only ten days ago ...... she started her new job.\n (a) then\n (b) since\n (c) after\n (d) that",
     "15. The shop didn't have the shoes I wanted, but they've ...... a pair specially for me.\n (a) booked\n (b) ordered\n (c) commanded\n (d) asked",
     "16. Have you got time to discuss your work now or are you ...... to leave?\n (a) thinking\n (b) round\n (c) planned\n (d) about",
     "17. She came to live here ...... a month ago.\n (a) quite\n (b) beyond\n (c) already\n (d) almost",
     "18. Once the plane is in the air, you can ...... your seat belts if you wish.\n (a) undress\n (b) unfasten\n (c) unlock\n (d) untie",
     "19. I left my last job because I had no ...... to travel.\n (a) place\n (b) position\n (c) opportunity\n (d) possibility",
     "20. It wasn't a bad crash and ...... damage was done to my car.\n (a) little\n (b) small\n (c) light\n (d) mere",
     "21. I'd rather you ...... to her why we can't go.\n (a) would explain\n (b) explained\n (c) to explain\n (d) will explain",
     "22. Before making a decision, the leader considered all ...... of the argument.\n (a) sides\n (b) features\n (c) perspectives\n (d) shades",
     "23. This new printer is recommended as being ...... reliable.\n (a) greatly\n (b) highly\n (c) strongly\n (d) readily",
     "24. When I realised I had dropped my gloves, I decided to ...... my steps.\n (a) retrace\n (b) regress\n (c) resume\n (d) return",
     "25. Anne's house is somewhere in the ...... of the railway station.\n (a) region\n (b) quarter\n (c) vicinity\n (d) district"]

questions = [
     Question(question_prompts[0], "c"),
     Question(question_prompts[1], "b"),
     Question(question_prompts[2], "a"),
     Question(question_prompts[3], "c"),
     Question(question_prompts[4], "a"),
     Question(question_prompts[5], "c"),
     Question(question_prompts[6], "b"),
     Question(question_prompts[7], "b"),
     Question(question_prompts[8], "c"),
     Question(question_prompts[9], "b"),
     Question(question_prompts[10],"d"),
     Question(question_prompts[11],"c"),
     Question(question_prompts[12],"b"),
     Question(question_prompts[13],"d"),
     Question(question_prompts[14],"b"),
     Question(question_prompts[15],"d"),
     Question(question_prompts[16],"d"),
     Question(question_prompts[17],"b"),
     Question(question_prompts[18],"c"),
     Question(question_prompts[19],"a"),
     Question(question_prompts[20],"b"),
     Question(question_prompts[21],"a"),
     Question(question_prompts[22],"b"),
     Question(question_prompts[23],"a"),
     Question(question_prompts[24],"c")]

def simple(questions):
    list1 = []
    list2 = list(questions)
    while len(list1)<25:
        elem = random.choice(list2)
        list2.remove(elem)
        list1.append(elem)
    return list1

def find_level(score):
    if score <= 5:
        return "A1"
    elif score > 5 and score <= 7:
        return "A2"
    elif score > 7 and score <= 10:
        return "B1"
    elif score >10 and score <= 15:
        return "B2"
    elif score >15 and score <= 20:
        return "C1"
    elif score >20 and score <= 25:
        return "C2"



@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content == "يا بانكو" :
        await message.channel.send('يا نعم')

    elif message.content.startswith("يا بانكو"):
        print(message.content)
        result_class = class_type(message.content[10:])
        if result_class == "take exam":
            await message.channel.send("لو اختارت حرف غير حروف الاختياري, هتتحسب خطأ")
            score = 0
            random_list = simple(questions)
            for question in random_list:
                await message.channel.send(question.prompt)
                answer = await client.wait_for("message")
                n = answer.content
                if n == question.answer:
                    score += 1
                    await message.channel.send("Correct")
                else:
                    await message.channel.send("Incorrect")
            response = f"نتيجتك هي {score}, وبالتالي فان مستواك هو {find_level(score)}"
        elif result_class == "peer grouping":
            await message.channel.send("بالرجاء اخيار مستواك.\nA1\nA2\nB1\nB2\nC1\nC2")
            msg = await client.wait_for("message")
            n = msg.content
            response = peer_grouping(n)

        elif result_class == "course recommendation":
            await message.channel.send("بالرجاء اخيار مستواك.\nA1\nA2\nB1\nB2\nC1\nC2")
            msg = await client.wait_for("message")
            n = msg.content
            response = course_recommendation(n)

        else:
            response = chat(message.content[10:])
        #await message.channel.send("response")
        await message.channel.send(response)

client.run(TOKEN)

