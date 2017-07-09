using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;


namespace SupportResistance
{
    class ModelData
    {
        /// <summary>
        /// The list of distributions for the price levels.
        /// </summary>
        public Gaussian[] PriceLevelDist;

        /// <summary>
        /// The list of distributions for noises.
        /// </summary>
        public Gamma[] NoiseDist;

        /// <summary>
        /// Gaussian mixture coefficients. 
        /// </summary>
        public Dirichlet Mixing;
    }

    class SupportResistanceBase
    {
        public InferenceEngine InferenceEngine;
        /// <summary>
        /// The number of components in gaussian mixture.
        /// </summary>
        public int NumComponents;

        /// <summary>
        /// Random variables representing the price levels for support/resistance lines.
        /// </summary>
        public VariableArray<double> PriceLevels;

        /// <summary>
        /// Random variables representing the noise around the price levels.
        /// </summary>
        public VariableArray<double> Noises;

        public VariableArray<Gaussian> PriceLevelPriors;
        public VariableArray<Gamma> NoisesPriors;

        public Variable<Dirichlet> MixingPrior;
        public Variable<Vector> MixingCoefficients;

        public virtual void CreateModel(int n) {
            InferenceEngine = new InferenceEngine(new VariationalMessagePassing());
            InferenceEngine.ShowFactorGraph = true;
            NumComponents = n;
            Range ComponentRange = new Range(NumComponents).Named("component_index");

            PriceLevelPriors = Variable.Array<Gaussian>(ComponentRange).Named("price_line_priors");
            NoisesPriors = Variable.Array<Gamma>(ComponentRange).Named("noise_priors");
            PriceLevels = Variable.Array<double>(ComponentRange).Named("price_lines");
            Noises = Variable.Array<double>(ComponentRange).Named("noises");

            // Iterate over each component in the gaussian mixture model.
            using (Variable.ForEach(ComponentRange))
            {
                // Each price level, that is, a support/resistance line is modeled as a Gaussian
                // whose mean and precision are from two prior distributions.
                // The prior of the mean is a Gaussian, and the prior of the precision is a Gamma distribution.
                PriceLevels[ComponentRange] = Variable.Random<double, Gaussian>(PriceLevelPriors[ComponentRange]);
                Noises[ComponentRange] = Variable.Random<double, Gamma>(NoisesPriors[ComponentRange]);
            }

            //Mixing coefficients
            MixingPrior = Variable.New<Dirichlet>().Named("price_line_mixture_prior");
            MixingCoefficients = Variable<Vector>.Random(MixingPrior).Named("mixture_coefficients");
            MixingCoefficients.SetValueRange(ComponentRange);
        }

        public virtual void SetModelData(ModelData modelData)
        {
            PriceLevelPriors.ObservedValue = modelData.PriceLevelDist;
            NoisesPriors.ObservedValue = modelData.NoiseDist;
            MixingPrior.ObservedValue = modelData.Mixing;
        }
    }

    class SupportResistanceTraining : SupportResistanceBase
    {
        Variable<int> NumPoints;
        VariableArray<double> Prices;
        VariableArray<int> ComponentIndices;

        /// <summary>
        /// Create the model.
        /// </summary>
        /// <param name="n">The number of data points in training data.</param>
        public override void CreateModel(int n)
        {            
            base.CreateModel(n);
            NumPoints = Variable.New<int>();
            Range dataPointRange = new Range(NumPoints).Named("sample_index");
            Prices = Variable.Array<double>(dataPointRange).Named("price_line_posteriors");
            ComponentIndices = Variable.Array<int>(dataPointRange).Named("price_line_mixture");

            // Iterate over each training data point.
            using (Variable.ForEach(dataPointRange))
            {
                // Iterate over each component in the Gaussian mixture.
                ComponentIndices[dataPointRange] = Variable.Discrete(MixingCoefficients);
                using (Variable.Switch(ComponentIndices[dataPointRange]))
                {
                    // Every component in the Gaussian mixture is a support/resistance line.
                    // That line is represented by a Gaussian variable whose mean and precision
                    // are from two prior distributions.
                    var p = PriceLevels[ComponentIndices[dataPointRange]];
                    var nz = Noises[ComponentIndices[dataPointRange]]; 
                    var line = Variable.GaussianFromMeanAndPrecision(p, nz);
                    Prices[dataPointRange].SetTo(line);                 
                }
            }
        }

        public ModelData InferModelData(double[] trainingData)
        {
            ModelData posteriors = new ModelData();
            NumPoints.ObservedValue = trainingData.Length;
            Prices.ObservedValue = trainingData;
            posteriors.PriceLevelDist = InferenceEngine.Infer<Gaussian[]>(PriceLevels);
            posteriors.NoiseDist = InferenceEngine.Infer<Gamma[]>(Noises);
            posteriors.Mixing = InferenceEngine.Infer<Dirichlet>(MixingCoefficients);
            return posteriors;
        }
    }
}
