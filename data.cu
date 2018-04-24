#include <stdio.h>

int main()
  {
  long  len;
  float* buf = NULL;
  FILE* fp  = NULL;

  // Open the source file
  fp = fopen("cifar-10-batches-bin/data_batch_1.bin", "rb" );
  if (!fp) return 0;

  // Get its length (in bytes)
  if (fseek( fp, 0, SEEK_END ) != 0)  // This should typically succeed
    {                                 // (beware the 2Gb limitation, though)
    fclose( fp );
    return 0;
    }

  len = ftell( fp );
  rewind( fp );

  // Get a buffer big enough to hold it entirely
  buf = (float*)malloc( len );
  if (!buf)
    {
    fclose( fp );
    return 0;
    }

  // Read the entire file into the buffer
  if (!fread( buf, len, 1, fp ))
    {
    free( buf );
    fclose( fp );
    return 0;
    }
	printf("%f\n",buf[0]);

  // All done -- return success
  fclose( fp );
  free( buf );
  return 1;
  }
