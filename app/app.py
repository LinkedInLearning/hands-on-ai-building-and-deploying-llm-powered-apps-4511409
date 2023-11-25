import chainlit as cl


##############################################################################
# Exercise 1a: 
# Please add the proper decorator to this main function so Chainlit will call
# this function when it receives a message
##############################################################################
async def main(message: cl.Message):

##############################################################################
# Exercise 1b:
# Please get the content of the chainlit Message and send it back as a
# response
##############################################################################
    await cl.Message().send()