/*
--------------------------------------------------
    91f7d09794d8da29f028e77df49d4907
--------------------------------------------------
    DaisyGAN

    This is kind of like my trading bots but
    for the weight_override was substituted
    for error_override as to feed back in a
    custom error value into the backprop.

    Only the arctan and lecun_tanh has good
    results, more so arctan hence having
    hardcoded the bot around this specific
    activator.
*/
/*
<?php
    $token = '<BOT TOKEN>';
    $j = json_decode(file_get_contents("php://input"));

    if(isset($j->{'message'}->{'text'}))
    {
        if(strstr($j->{'message'}->{'text'}, "/isdaisy") != FALSE && isset($j->{'message'}->{'reply_to_message'}->{'text'}))
        {
            $percent = rtrim(shell_exec('/srv/cfdgan "' . $j->{'message'}->{'reply_to_message'}->{'text'} . '"'), "\n");
            $chatid = $j->{'message'}->{'chat'}->{'id'};
            file_get_contents("https://api.telegram.org/bot" . $token . "/sendMessage?chat_id=" . $chatid . "&text=".urlencode('"'.$j->{'message'}->{'reply_to_message'}->{'text'}).'"'."+is+" . urlencode($percent) . "%25+something+Daisy+(@VXF97)+would+say.");
            http_response_code(200);
            exit;
        }

        if(strstr($j->{'message'}->{'text'}, "/quote") != FALSE)
        {
            $file = file("k.txt"); 
            $line = $file[rand(0, count($file) - 1)];
            $chatid = $j->{'message'}->{'chat'}->{'id'};
            file_get_contents("https://api.telegram.org/bot" . $token . "/sendMessage?chat_id=" . $chatid . "&text=".urlencode($line));
            http_response_code(200);
            exit;
        }
    }

    http_response_code(200);
?>
*/

#pragma GCC diagnostic ignored "-Wunused-result"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>

#define uint uint32_t

#define DIGEST_SIZE 256
#define HIDDEN_SIZE 8192
#define DATA_SIZE 1228

#define TRAINING_LOOPS 1

uint  _activator = 5;
uint  _optimiser = 0;
float _lrate = 0.003;
float _lmomentum = 0.9;
float _dropout = 0.5; //chance of neuron droput 0.3 = 30%

struct
{
    float* data;
    float* momentum;
    float bias;
    float bias_momentum;
    uint weights;
}
typedef ptron;

// discriminator 
ptron d1[DIGEST_SIZE];
ptron d2[HIDDEN_SIZE];
ptron d3;

// generator
ptron g1[DIGEST_SIZE];
ptron g2[HIDDEN_SIZE];
ptron g3[DIGEST_SIZE];

// normalised training data
float digest[DATA_SIZE][DIGEST_SIZE] = {0};


//*************************************
// utility functions
//*************************************

void saveWeights()
{
    FILE* f = fopen("weights.dat", "w");
    if(f != NULL)
    {
        for(uint i = 0; i < DIGEST_SIZE; i++)
        {
            if(fwrite(&d1[i].data[0], 1, d1[i].weights*sizeof(float), f) != d1[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1w\n");
            
            if(fwrite(&d1[i].momentum[0], 1, d1[i].weights*sizeof(float), f) != d1[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1m\n");

            if(fwrite(&d1[i].bias, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1w\n");
            
            if(fwrite(&d1[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1m\n");
        }

        for(uint i = 0; i < HIDDEN_SIZE; i++)
        {
            if(fwrite(&d2[i].data[0], 1, d2[i].weights*sizeof(float), f) != d2[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1w\n");
            
            if(fwrite(&d2[i].momentum[0], 1, d2[i].weights*sizeof(float), f) != d2[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1m\n");

            if(fwrite(&d2[i].bias, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1w\n");
            
            if(fwrite(&d2[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1m\n");
        }

        if(fwrite(&d3.data[0], 1, d3.weights*sizeof(float), f) != d3.weights*sizeof(float))
            printf("ERROR fwrite() in saveWeights() #1w\n");
        
        if(fwrite(&d3.momentum[0], 1, d3.weights*sizeof(float), f) != d3.weights*sizeof(float))
            printf("ERROR fwrite() in saveWeights() #1m\n");

        if(fwrite(&d3.bias, 1, sizeof(float), f) != sizeof(float))
            printf("ERROR fwrite() in saveWeights() #1w\n");
        
        if(fwrite(&d3.bias_momentum, 1, sizeof(float), f) != sizeof(float))
            printf("ERROR fwrite() in saveWeights() #1m\n");

        fclose(f);
    }
}

void loadWeights()
{
    FILE* f = fopen("weights.dat", "r");
    if(f == NULL)
    {
        printf("!!! no pre-existing weights where found, starting from random initialisation.\n\n\n-----------------\n");
        return;
    }

    for(uint i = 0; i < DIGEST_SIZE; i++)
    {
        while(fread(&d1[i].data[0], 1, d1[i].weights*sizeof(float), f) != d1[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d1[i].momentum[0], 1, d1[i].weights*sizeof(float), f) != d1[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d1[i].bias, 1, sizeof(float), f) != sizeof(float))
            sleep(333);

        while(fread(&d1[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
            sleep(333);
    }

    for(uint i = 0; i < HIDDEN_SIZE; i++)
    {
        while(fread(&d2[i].data[0], 1, d2[i].weights*sizeof(float), f) != d2[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d2[i].momentum[0], 1, d2[i].weights*sizeof(float), f) != d2[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d2[i].bias, 1, sizeof(float), f) != sizeof(float))
            sleep(333);

        while(fread(&d2[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
            sleep(333);
    }

    while(fread(&d3.data[0], 1, d3.weights*sizeof(float), f) != d3.weights*sizeof(float))
            sleep(333);

    while(fread(&d3.momentum[0], 1, d3.weights*sizeof(float), f) != d3.weights*sizeof(float))
            sleep(333);

    while(fread(&d3.bias, 1, sizeof(float), f) != sizeof(float))
            sleep(333);

    while(fread(&d3.bias_momentum, 1, sizeof(float), f) != sizeof(float))
        sleep(333);

    fclose(f);
}

float qRandWeight(const float min, const float max)
{
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srand(time(0));
        ls = time(0) + 33;
    }
    float pr = 0;
    while(pr == 0) //never return 0
    {
        const float rv = (float)rand();
        if(rv == 0)
            return min;
        const float rv2 = ( (rv / RAND_MAX) * (max-min) ) + min;
        pr = roundf(rv2 * 100) / 100; // two decimals of precision
    }
    return pr;
}

float uRandWeight(const float min, const float max)
{
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint s = 0;
    ssize_t result = read(f, &s, 4);
    srand(s);
    close(f);
    float pr = 0;
    while(pr == 0) //never return 0
    {
        const float rv = (float)rand();
        if(rv == 0)
            return min;
        const float rv2 = ( (rv / RAND_MAX) * (max-min) ) + min;
        pr = roundf(rv2 * 100) / 100; // two decimals of precision
    }
    return pr;
}

uint qRand(const uint min, const uint umax)
{
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srand(time(0));
        ls = time(0) + 33;
    }
    const int rv = rand();
    const uint max = umax + 1;
    if(rv == 0)
        return min;
    return ( ((float)rv / RAND_MAX) * (max-min) ) + min; //(rand()%(max-min))+min;
}

uint uRand(const uint min, const uint umax)
{
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint s = 0;
    ssize_t result = read(f, &s, 4);
    srand(s);
    close(f);
    const int rv = rand();
    const uint max = umax + 1;
    if(rv == 0)
        return min;
    return ( ((float)rv / RAND_MAX) * (max-min) ) + min; //(rand()%(max-min))+min;
}



//*************************************
// create layer
//*************************************

void createPerceptron(ptron* p, const uint weights, const uint random)
{
    p->data = malloc(weights * sizeof(float));
    if(p->data == NULL)
    {
        printf("Perceptron creation failed (w)%u.\n", weights);
        return;
    }

    p->momentum = malloc(weights * sizeof(float));
    if(p->momentum == NULL)
    {
        printf("Perceptron creation failed (m)%u.\n", weights);
        return;
    }

    p->weights = weights;

    if(random == 1)
    {
        for(uint i = 0; i < weights; i++)
        {
            p->data[i] = qRandWeight(-1, 1);
            p->momentum[i] = 0;
        }
    }
    else if(random == 0)
    {
        memset(p->data, 0, weights * sizeof(float));
        memset(p->momentum, 0, weights * sizeof(float));
    }

    p->bias = qRandWeight(-1, 1);
    p->bias_momentum = 0;
}


//*************************************
// activation functions
// https://en.wikipedia.org/wiki/Activation_function
// https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/
// https://adl1995.github.io/an-overview-of-activation-functions-used-in-neural-networks.html
//*************************************

static inline float bipolarSigmoid(float x)
{
    return (1 - exp(-x)) / (1 + exp(-x));
}

static inline float fbiSigmoid(float x)
{
    return (1 - fabs(x)) / (1 + fabs(x));
}

static inline float arctan(float x)
{
    return atan(x);
}

static inline float lecun_tanh(float x)
{
    return 1.7159 * atan(0.666666667 * x);
}

static inline float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

static inline float fSigmoid(float x)
{
    return x / (1 + fabs(x));
}

static inline float swish(float x)
{
    return x * sigmoid(x);
}

static inline float leakyReLU(float x)
{
    if(x < 0){x *= 0.01;}
    return x;
}

static inline float ReLU(float x)
{
    if(x < 0){x = 0;}
    return x;
}

static inline float ReLU6(float x)
{
    if(x < 0){x = 0;}
    if(x > 6){x = 6;}
    return x;
}

static inline float leakyReLU6(float x)
{
    if(x < 0){x *= 0.01;}
    if(x > 6){x = 6;}
    return x;
}

static inline float smoothReLU(float x)
{
    return log(1 + exp(x));
}

static inline float logit(float x)
{
    return log(x / (1 - x));
}

void softmax_transform(float* w, const uint32_t n)
{
    float d = 0;
    for(size_t i = 0; i < n; i++)
        d += exp(w[i]);

    for(size_t i = 0; i < n; i++)
        w[i] = exp(w[i]) / d;
}

float crossEntropy(const float predicted, const float expected) //log loss
{
    if(expected == 1)
      return -log(predicted);
    else
      return -log(1 - predicted);
}


//*************************************
// neural net / perceptron funcs
//*************************************

float doPerceptron(const float* in, ptron* p, const float error_override, const float eo)
{
//~~ Perceptron dropout [only during training]
    if(eo != -2 || error_override != 0)
    {
        if(_dropout != 0)
        {
            if(uRandWeight(0.01, 1) <= _dropout)
                return 0;
        }
    }
    
//~~ Query perceptron
    //Sum inputs mutliplied by weights
    float ro = 0;
    for(uint i = 0; i < p->weights; i++)
        ro += in[i] * p->data[i];
    ro += p->bias;

//~~ Activation Function
    if(_activator == 1)
        ro = sigmoid(ro);
    else if(_activator == 2)
    {
        if(ro < 0){ro = 0;}
    }
    else if(_activator == 3)
        ro = leakyReLU(ro);
    else if(_activator == 4)
        ro = smoothReLU(ro);
    else if(_activator == 5)
        ro = arctan(ro);
    else if(_activator == 6)
        ro = tanh(ro);
    else if(_activator == 7)
        ro = lecun_tanh(ro);
    else if(_activator == 8)
        ro = logit(ro);
    else if(_activator == 9)
        ro = bipolarSigmoid(ro);
    else if(_activator == 10)
        ro = fbiSigmoid(ro);
    else if(_activator == 11)
        ro = fSigmoid(ro);
    else if(_activator == 12)
        ro = swish(ro);
    else if(_activator == 13)
        ro = leakyReLU6(ro);
    else if(_activator == 14)
        ro = ReLU6(ro);

//~~ Teach perceptron
    if(eo != -2 || error_override != 0)
    {
        const float error = error_override == 0 ? eo - ro : error_override - ro; // error gradient
        for(uint i = 0; i < p->weights; i++)
        {
            // Regular Gradient Descent
            if(_optimiser == 0)
                p->data[i] += error * in[i] * _lrate;

            // Regular Momentum
            if(_optimiser == 1)
            {
                p->momentum[i] = _lmomentum * p->momentum[i] + (error * in[i] * _lrate);
                p->data[i] += p->momentum[i];
            }
            
            // Nesterov (NAG) momentum
            if(_optimiser == 2)
            {
                const float g = error * in[i] * _lrate;
                const float l = p->data[i] + p->momentum[i];
                p->momentum[i] = _lmomentum * p->momentum[i] + g * l;
                p->data[i] += p->momentum[i];
            }
        }

        // Regular Gradient Descent
        if(_optimiser == 0)
            p->bias += error * _lrate;

        // Regular Momentum
        if(_optimiser == 1)
        {
            p->bias_momentum = _lmomentum * p->bias_momentum + (error * _lrate);
            p->bias += p->bias_momentum;
        }
        
        // Nesterov (NAG) momentum
        if(_optimiser == 2)
        {
            p->bias_momentum = _lmomentum * p->bias_momentum + (error * _lrate) * (p->bias + p->bias_momentum);
            p->bias += p->bias_momentum;
        }
    }

//~~ Return output
    return ro;
}


//*************************************
// network training functions
//*************************************

float doDiscriminator(const float* input, const float eo)
{
    // layer one, inputs (fc)
    float o1[DIGEST_SIZE];
    for(int i = 0; i < DIGEST_SIZE; i++)
        o1[i] = doPerceptron(input, &d1[i], 0, eo);

    // layer two, hidden (fc expansion)
    float o2[HIDDEN_SIZE];
    for(int i = 0; i < HIDDEN_SIZE; i++)
        o2[i] = doPerceptron(&o1[0], &d2[i], 0, eo);
    
    // layer three, output (fc compression)
    return doPerceptron(&o2[0], &d3, 0, eo);
}

void doGenerator(const float error, const float* input, float* output)
{
    // layer one, inputs (fc)
    float o1[DIGEST_SIZE];
    for(int i = 0; i < DIGEST_SIZE; i++)
        o1[i] = doPerceptron(input, &g1[i], error, -2);

    // layer two, hidden (fc expansion)
    float o2[HIDDEN_SIZE];
    for(int i = 0; i < HIDDEN_SIZE; i++)
        o2[i] = doPerceptron(&o1[0], &g2[i], error, -2);
    
    // layer three, output (fc compression)
    for(int i = 0; i < DIGEST_SIZE; i++)
        output[i] = doPerceptron(&o2[0], &g3[i], error, -2);
}

/*
--------------------------------------------------
    Defunct Backprop Method Start
--------------------------------------------------
*/
void doSGD(float* weight, float* momentum, const float error)
{
    // Regular Gradient Descent
    if(_optimiser == 0)
    {
        weight[0] += error * _lrate;
    }

    // Regular Momentum
    if(_optimiser == 1)
    {
        momentum[0] = _lmomentum * momentum[0] + (error * _lrate);
        weight[0] += *momentum;
    }
    
    // Nesterov (NAG) momentum
    if(_optimiser == 2)
    {
        momentum[0] = _lmomentum * momentum[0] + (error * _lrate) * (weight[0] + momentum[0]);
        weight[0] += momentum[0];
    }
}

void backpropGenerator(const float error)
{
    // layer one, inputs (fc)
    for(int i = 0; i < DIGEST_SIZE; i++)
    {
        const float layer_error = (1-(1/i))*error;

        for(int j = 0; j < g1[i].weights; j++)
            doSGD(&g1[i].data[j], &g1[i].momentum[j], layer_error);
        
        doSGD(&g1[i].bias, &g1[i].bias_momentum, layer_error);
    }

    // layer two, hidden (fc expansion)
    for(int i = 0; i < HIDDEN_SIZE; i++)
    {
        const float layer_error = (1-(1/i))*error;

        for(int j = 0; j < g2[i].weights; j++)
            doSGD(&g2[i].data[j], &g2[i].momentum[j], layer_error);
        
        doSGD(&g2[i].bias, &g2[i].bias_momentum, layer_error);
    }
    
    // layer three, output (fc compression)
    for(int i = 0; i < DIGEST_SIZE; i++)
    {
        const float layer_error = (1-(1/i))*error;

        for(int j = 0; j < g3[i].weights; j++)
            doSGD(&g3[i].data[j], &g3[i].momentum[j], layer_error);
        
        doSGD(&g3[i].bias, &g3[i].bias_momentum, layer_error);
    }
}
/*
--------------------------------------------------
    Defunct Backprop Method End
--------------------------------------------------
*/

float rmseDiscriminator()
{
    float squaremean = 0;
    for(int i = 0; i < DATA_SIZE; i++)
    {
        const float r = 1 - doDiscriminator(&digest[i][0], -2);
        squaremean += r*r;
    }
    squaremean /= DATA_SIZE;
    return sqrt(squaremean);
}

void trainDataset(const char* file)
{
    // read training data [every input is truncated to 256 characters]
    FILE* f = fopen(file, "r");
    if(f)
    {
        char line[DIGEST_SIZE];
        float nline[DIGEST_SIZE];
        uint index = 0;
        while(fgets(line, DIGEST_SIZE, f) != NULL)
        {
            // normalise line data
            for(int i = 0; i < DIGEST_SIZE; i++)
                digest[index][i] = line[i] == 0 ? 0 : ( ((double)line[i]) / 128.0 ) - 1.0;

            index++;
        }
        fclose(f);
    }

    printf("Training Data Loaded\n");

    // train discriminator
    for(int j = 0; j < TRAINING_LOOPS; j++)
    {
        for(int i = 0; i < DATA_SIZE; i++)
        {
            // train discriminator on data
            doDiscriminator(&digest[i][0], 1.57079632679);

            // detrain discriminator on random data
            float input[DIGEST_SIZE] = {0};
            const uint len = uRand(1, DIGEST_SIZE-1);
            for(int i = 0; i < len; i++)
                input[i] = uRandWeight(-1, 1);
            doDiscriminator(&input[0], -1.57079632679);

            printf("Training Iteration (%u / %u) [%u / %u]\n", i+1, DATA_SIZE, j+1, TRAINING_LOOPS);
        }

        printf("RMSE: %f\n", rmseDiscriminator());
    }

    // save weights
    saveWeights();
    printf("Weights Saved: weights.dat\n");
}

void consoleAsk()
{
    // what percentage daisy is this ?
    while(1)
    {
        char str[DIGEST_SIZE] = {0};
        float nstr[DIGEST_SIZE] = {0};
        printf(": ");
        fgets(str, DIGEST_SIZE, stdin);

        // normalise
        for(int i = 0; i < DIGEST_SIZE; i++)
            nstr[i] = str[i] == 0 ? 0 : ( ((double)str[i]) / 128.0 ) - 1.0;

        const float r = doDiscriminator(nstr, -2);
        printf("This is %.2f%% (%.2f) Daisy.\n", (r+1.57079632679)*31.830988618, r); //arctan conversion
    }
}

float isDaisy(const char* str)
{
    float nstr[DIGEST_SIZE] = {0};

    //normalise
    for(int i = 0; i < DIGEST_SIZE; i++)
        nstr[i] = str[i] == 0 ? 0 : ( ((double)str[i]) / 128.0 ) - 1.0;

    const float r = doDiscriminator(nstr, -2);
    return (r+1.57079632679)*31.830988618; //arctan conversion
}

float rndDaisy()
{
    float nstr[DIGEST_SIZE] = {0};
    for(int i = 0; i < DIGEST_SIZE; i++)
        nstr[i] = uRandWeight(-1, 1);

    const float r = doDiscriminator(nstr, -2);
    return (r+1.57079632679)*31.830988618; //arctan conversion
}


//*************************************
// program entry point
//*************************************

int main(int argc, char *argv[])
{
    // init discriminator
    for(int i = 0; i < DIGEST_SIZE; i++)
        createPerceptron(&d1[i], DIGEST_SIZE, 1);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        createPerceptron(&d2[i], DIGEST_SIZE, 1);
    createPerceptron(&d3, HIDDEN_SIZE, 1);

    // init generator
    for(int i = 0; i < DIGEST_SIZE; i++)
        createPerceptron(&g1[i], DIGEST_SIZE, 1);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        createPerceptron(&g2[i], DIGEST_SIZE, 1);
    for(int i = 0; i < DIGEST_SIZE; i++)
        createPerceptron(&g3[i], HIDDEN_SIZE, 1);

    // are we issuing any commands?
    if(argc == 3)
    {
        if(strcmp(argv[1], "relearn") == 0)
        {
            trainDataset(argv[2]);
            exit(0);
        }
    }

    if(argc == 2)
    {
        if(strcmp(argv[1], "retrain") == 0)
        {
            trainDataset("kds.txt");
            exit(0);
        }

        loadWeights();

        if(strcmp(argv[1], "ask") == 0)
            consoleAsk();
        
        if(strcmp(argv[1], "rnd") == 0)
        {
            printf("> %.2f\n", rndDaisy());
            exit(0);
        }

        if(strcmp(argv[1], "rndloop") == 0)
        {
            while(1)
            {
                printf("> %.2f\n", rndDaisy());
                //usleep(100000);
            }
        }

        char in[DIGEST_SIZE] = {0};
        snprintf(in, DIGEST_SIZE, "%s", argv[1]);
        printf("%.2f\n", isDaisy(in));
        exit(0);
    }

    // no commands ? then I would suppose we are running the generator
    loadWeights();

    // train generator
    uint index = 0;
    float last_error = 0;
    while(1)
    {
        // random generator input
        float input[DIGEST_SIZE] = {0};
        for(int i = 0; i < DIGEST_SIZE; i++)
            input[i] = uRandWeight(-1, 1);

        // do generator
        float output[DIGEST_SIZE] = {0};
        doGenerator(last_error * 0.3, &input[0], &output[0]);

        // convert output to string
        char str[DIGEST_SIZE+1] = {0};
        for(int i = 0; i < DIGEST_SIZE; i++)
        {
            str[i] = (output[i]+1.57079632679)*81.169020977; //arctan conversion
            //printf("^%.2f\n", (output[i]+1.57079632679)*81.169020977);
        }
        str[DIGEST_SIZE] = 0x00;
        printf("%s\n", str);

        // feed generator output into discriminator input, take the error, sigmoid it to 0-1, take the loss, put it back through as the error for the next generation
        last_error = crossEntropy(sigmoid(3.141592654 - (doDiscriminator(&output[0], -2) + 1.57079632679)), 1);

        // back prop the generator [defunct method of backprop]
        //backpropGenerator(last_error);

        // output error
        printf("[%u]ERROR: %.2f\n\n", index, last_error);
        index++;
        //sleep(1);
    }

    // done
    return 0;
}

