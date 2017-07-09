using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using System.IO;



namespace SupportResistance
{   
    class Program
    {
        static void Main(string[] args)
        {
            // Command line arguments.
            // First argument: path to the training data file, whose content is a comma-separated prive values.
            string f = args[0];

            // Second argument: path to the file to store trained models.
            string outputFile = args[1];
            double[] trainingData = null;
            using (var sr = new StreamReader(f))
            {
                var numbers = sr.ReadToEnd().Split(',');
                trainingData = new double[numbers.Length];
                int i = 0;
                foreach(var n in numbers) {
                    trainingData[i] = Convert.ToDouble(n);
                    i++;
                }
            }            
            double maxPrice = trainingData.Max();
            double minPrice = trainingData.Min();

            // Third command line argument: the number of components in the Gaussian mixture model.
            int minComp = Convert.ToInt32(args[2]);
            int maxComp = minComp;

            // Create priors for the distributions to be learnt.
            SupportResistanceTraining p = new SupportResistanceTraining();
            Dictionary<int, double> evidences = new Dictionary<int, double>();
            for (int mixNum = minComp; mixNum <= maxComp; mixNum++)
            {
                double stepPrice = (maxPrice - minPrice) / (double)mixNum;

                double[] dd = new double[mixNum];
                ModelData initPriors = new ModelData();
                initPriors.PriceLevelDist = new Gaussian[mixNum];
                initPriors.NoiseDist = new Gamma[mixNum];
                for (int i = 0; i < mixNum; i++)
                {
                    initPriors.PriceLevelDist[i] = new Gaussian(minPrice + i * stepPrice, 1.0);
                    initPriors.NoiseDist[i] = new Gamma(2.0, 2.0);
                    dd[i] = 1.0 / (double)mixNum;
                }
                double s = 0;
                for (int i = 0; i < mixNum - 1; i++)
                {
                    s += dd[i];
                }
                dd[mixNum - 1] = 1.0 - s;
                initPriors.Mixing = new Dirichlet(dd);

                // Train model and get inferred posteriors.
                p.CreateModel(mixNum);
                p.SetModelData(initPriors);
                var posteriors = p.InferModelData(trainingData);

                // Write model into output file.
                Console.Write("[");
                for (int i = 0; i < mixNum; i++)
                {
                    Console.Write(posteriors.PriceLevelDist[i].GetMean());
                    if (i < mixNum - 1)
                    {
                        Console.Write(", ");
                    }
                }
                Console.WriteLine("]");

                Console.Write("Precision:\n[");
                for (int i = 0; i < mixNum; i++)
                {
                    Console.Write(posteriors.PriceLevelDist[i].GetVariance());
                    if (i < mixNum - 1)
                    {
                        Console.Write(", ");
                    }
                }
                Console.WriteLine("]");

                using (var sw = new StreamWriter(outputFile))
                {
                    for (int i = 0; i < mixNum; i++)
                    {
                        sw.Write(posteriors.PriceLevelDist[i].GetMean());
                        if (i < mixNum - 1)
                        {
                            sw.Write(",");
                        }
                    }
                    sw.WriteLine();

                    for (int i = 0; i < mixNum; i++)
                    {
                        sw.Write(posteriors.PriceLevelDist[i].GetVariance());
                        if (i < mixNum - 1)
                        {
                            sw.Write(",");
                        }
                    }                    
                    sw.WriteLine();
                    for (int i = 0; i < mixNum; i++)
                    {
                        sw.Write(posteriors.Mixing.GetMean()[i]);
                        if (i < mixNum - 1)
                        {
                            sw.Write(",");
                        }
                    }
                }
            }
        }
    }
}
