#include <iostream>
#include <fstream>
#include <stdlib.h> 

int
  main (int argc, char** argv)
{

  std::ifstream infile(argv[1]);
std::cout<<argv[1]<<std::endl;

    std::ofstream fpoints;
    fpoints.open("test_xyz.xyz");
    fpoints << std::fixed;

float x,y,z;
long c;
int i=0;
std::string header;
int count = 47;
  while (i<count || infile >> x >> y >> z >> c)
  {
    if(i<count){
std::cout<<i<<std::endl;
infile>>header;
std::cout<<header<<std::endl;
i++;
continue;
}
    fpoints << x
                << " " << y
                << " " << z
                << std::endl;
  }
fpoints.close();

  return (0);
}
