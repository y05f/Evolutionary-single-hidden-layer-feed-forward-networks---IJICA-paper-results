//2013-12-22 01:44:13
#include "EMlp.cpp"
using namespace std;

int main()
{

  //system("rm -fr ./save/*");
  //system("rm -fr ./graph/*");
  int cm = 2;
  cout << "0(NewEmlp), 1(testMlp)? :";
  cin >> cm;
  if (cm == 0)
    Construct(30);
  else if (cm == 1)
    TestConstruct(30);
  else
    exit(0);
  return 0;
}
