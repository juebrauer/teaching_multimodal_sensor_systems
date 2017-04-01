#include <conio.h> // _getch()
#include <stdio.h> // printf()
#include <stdlib.h> // rand()
#include <time.h>   // time()
#include <vector>   // for data container std::vector

using namespace std;

// base class
class RoadUser
{

public:
   RoadUser()
   {
      printf("Constructor of base class RoadUser called.\n");
      relx = rand() % 101;
      rely = rand() % 101;
   }

   virtual void show_data()
   {
      printf(" relx=%.1f, rely=%.1f, absv=%.1f\n",
         relx, rely, absv); 
   }
   
protected:
   double relx;
   double rely;
   double absv;

};

// a first derived class
class Pedestrian : public RoadUser
{
   public:
   Pedestrian()
   {
      printf("New Pedestrian object generated\n"); 
      absv = rand() % 11;
   }

   ~Pedestrian()
   {
      printf("Destructor of Pedestrian called.\n");
   }

   void show_data()
   {
      printf("Pedestrian: ");
      RoadUser::show_data();
   }

};



class Car : public RoadUser
{
public:
   Car()
   {
      printf("New car object generated\n");
      absv = rand() % 251;
   }

   ~Car()
   {
      printf("Destructor of car called.\n");
   }

   void show_data()
   {
      printf("Car: ");
      RoadUser::show_data();
   }
 
};


int main()
{
   // initialize pseudo-random number generator
   srand( (unsigned int) time(NULL) ); 

   printf("ObjectListSimulator was started.\n");

   int simstep=0;
   vector<RoadUser*> roaduserlist; 

   while (true)
   {       
      simstep++;
      printf("\nSimulation step: %d\n", simstep);

      int rndnr = rand() % 2;

      if (rndnr == 0)
      {
         printf("Generating new object\n");
         int rndnr2 = rand() % 2;
         RoadUser* r;
         if (rndnr2==0)
            r = new Pedestrian();
         else
            r = new Car();
         roaduserlist.push_back( r );
      }
      else
      {
         printf("Deleting existing object\n");
         int N = roaduserlist.size();
         if (N > 0)
         {
            // delete object from memory
            RoadUser* ptr = roaduserlist.at(0);
            delete ptr;

            // delete pointer to object from vector
            roaduserlist.erase(
               roaduserlist.begin() + 0);
         }
         else
         {
            printf("There is no object to delete!\n");
         }
      }

      // show list of road users
      printf("List of all road users:\n");
      int N = roaduserlist.size();
      for (int i = 0; i < N; i++)
      {
         RoadUser* r = roaduserlist.at( i );
         r->show_data();
      }

      _getch();
   }

   Pedestrian* p1 = new Pedestrian();
   Pedestrian* p2 = new Pedestrian();
   p1->show_data();
   p2->show_data();

}