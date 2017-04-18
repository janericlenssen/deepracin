/**
* \brief MaxPooling
* \param[in] gInputImage The input image
* \param[out] gOutputImage Filtered output image
* \param[in] width Width of the image
* \param[in] height Height of the image
* \param[in] depth Height of the image
* \param[in] sizex window size x-dim
* \param[in] sizey window size y-dim
* \param[in] sizez window size z-dim
* \param[in] strifex strife in x-dim
* \param[in] strifey strife in y-dim
* \param[in] strifez strife in z-dim
* \author jan
*/
__kernel void maxPool(
    const __global  float * gImage,
    __global  float * gOutputImage,
    const int width,
    const int height,
    const int depth,
    const int sizex,
    const int sizey,
    const int sizez,
    const int strifex,
    const int strifey,
    const int strifez
)
{
    const int indexx = get_global_id(0)*strifex;
    const int indexy = get_global_id(1)*strifey;
    const int indexz = get_global_id(2)*strifez;
    const int outputwidth = (int)ceil((float)width/strifex);
    const int outputheight= (int)ceil((float)width/strifey);
    const int ygap = width-sizex;
    const int zgap = width*(height-sizey)-sizex;

    //Implementation without using local memory. If overlapping is happening (stride<size) maybe inefficient.
    float maxf = 0.0;
    int index = mad24(indexz, width*height, mad24(indexy, width, indexx));
    for (int z = 0; z < sizez; z++)
    {
        for (int y = 0; y < sizey; y++)
        {
            for (int x = 0; x < sizex; x++)
            {
                    maxf = max(maxf,gImage[index]);
                    index++;
            }
            index+=ygap;
        }
        index+=zgap;
    }

    gOutputImage[get_global_id(2)*outputwidth*outputheight + get_global_id(1)*outputwidth + get_global_id(0)] = maxf;


}

/**
* \brief AvgPool
* \param[in] gInputImage The input image
* \param[out] gOutputImage Filtered output image
* \param[in] width Width of the image
* \param[in] height Height of the image
* \param[in] depth Height of the image
* \param[in] sizex window size x-dim
* \param[in] sizey window size y-dim
* \param[in] sizez window size z-dim
* \param[in] strifex strife in x-dim
* \param[in] strifey strife in y-dim
* \param[in] strifez strife in z-dim
* \author jan
*/
__kernel void avgPool(
    const __global  float * gImage,
    __global  float * gOutputImage,
    const int width,
    const int height,
    const int depth,
    const int sizex,
    const int sizey,
    const int sizez,
    const int strifex,
    const int strifey,
    const int strifez
)
{
    int indexx = get_global_id(0)*strifex;
    int indexy = get_global_id(1)*strifey;
    int indexz = get_global_id(2)*strifez;
    int outputwidth = (int)ceil((float)width/strifex);
    int outputheight= (int)ceil((float)width/strifey);

    //Implementation without using local memory. If overlapping is happening (stride<size) maybe inefficient.
    float averagef = 0.0;
    for (int zz = indexz; zz < indexz+sizez; zz++)
    {
        for (int yy = indexy; yy < indexy+sizey; yy++)
        {
            for (int xx = indexx; xx <= indexx+sizex; xx++)
            {
                    int index = mad24(zz, width*height, mad24(yy, width, xx));
                    averagef += gImage[index];
            }
        }
    }

    gOutputImage[get_global_id(2)*outputwidth*outputheight + get_global_id(1)*outputwidth + get_global_id(0)] = averagef/(sizex*sizey*sizez);


}
