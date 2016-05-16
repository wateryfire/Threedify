#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <fstream>
#include <stdlib.h> 

int
main (int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    std::cerr << argv[0] << std::endl;
    std::cerr << argv[1] << std::endl;
    std::cerr << argv[2] << std::endl;
    int count = atoi(argv[2]);
    std::cerr << count << std::endl;
    // Fill in the cloud data
    cloud.width    = count;
    cloud.height   = 1;
    cloud.is_dense = false;
    cloud.points.resize (cloud.width * cloud.height);

    std::ifstream infile(argv[1]);

    float a, b,x,y,z,R,G,B;
    size_t i = 0;

    if(argc==4){
        x=y=z=atof(argv[3]);
    }else if(argc==6){
        x=atof(argv[3]);
        y=atof(argv[4]);
        z=atof(argv[5]);
    }

    while (infile >> a >> b >> x >> y >> z >> R >> G >>B)
    {
        if(abs(x)>20|| abs(y)>20||abs(z)>20) continue;
        cloud.points[i].x = x;
        cloud.points[i].y = y;
        cloud.points[i].z = z;

        uint32_t rgb;
        rgb =   ( static_cast<int> ( B ) ) << 16 |
                    ( static_cast<int> ( G ) ) << 8 |
                    ( static_cast<int> ( R ) );
        cloud.points[i].rgb = rgb;
        i++;
    }

    if(cloud.empty()){
        std::cerr << "empty cloud file" << std::endl;
        return 1;
    }

    pcl::io::savePCDFileASCII ("test_pcd.pcd", cloud);
    std::cerr << "Saved " << cloud.points.size () << " data points to test_pcd.pcd." << std::endl;

    //for (size_t i = 0; i < cloud.points.size (); ++i)
    //  std::cerr << "    " << cloud.points[i].x << " " << cloud.points[i].y << " " << cloud.points[i].z << std::endl;

    return (0);
}
