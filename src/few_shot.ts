import { OpenAI } from "langchain/llms";
import * as dotenv from "dotenv";
import { LLMChain } from "langchain/chains";

  dotenv.config();


  import {
    LengthBasedExampleSelector,
    PromptTemplate,
    FewShotPromptTemplate,
  } from "langchain/prompts";

  
  export async function run() {
    const examplePrompt = new PromptTemplate({
      inputVariables: ["input", "output"],
      template: "Input: {input}\nOutput: {output}",
    });
  
    const exampleSelector = await LengthBasedExampleSelector.fromExamples(
      [
        { input: "happy", output: "sad" },
        { input: "tall", output: "short" },
        { input: "energetic", output: "lethargic" },
        { input: "sunny", output: "gloomy" },
        { input: "windy", output: "calm" },
      ],
      {
        examplePrompt,
        maxLength: 25,
      }
    );
  
    const dynamicPrompt = new FewShotPromptTemplate({
      exampleSelector,
      examplePrompt,
      prefix: "Give the antonym of every input",
      suffix: "Input: {adjective}\nOutput:",
      inputVariables: ["adjective"],
    });

    console.log(await dynamicPrompt.format({ adjective: "big" }));

    const longString =
      "big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else";
    const prompt = await dynamicPrompt.format({ adjective: longString });
    console.log(prompt);
    const model = new OpenAI({ temperature: 0.9 });
    
    const res = await model.call(prompt);

    console.log(res);

  }

run();