// MultiMedia_1.cpp : Defines the entry point for the application.
//
#include <windows.h>
#include <iostream>

#include <cstdint>
#include <EASTL/vector.h>

#include "D3DResources.h"
#include "ImageProcessor.h"

#ifdef _DEBUG
#pragma comment(linker, "/entry:wWinMainCRTStartup /subsystem:console")
#endif



#define MAX_LOADSTRING 100

// Global Variables:
HINSTANCE hInst;                                // current instance
WCHAR szTitle[MAX_LOADSTRING];                  // The title bar text
WCHAR szWindowClass[MAX_LOADSTRING];            // the main window class name
HWND hWnd;

D3DResources resource;
ImageProcessor imageProcess;



// Forward declarations of functions included in this code module:
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);

void SetObjectHandles(void);
void Update(void);
void Draw(void);
void CloseObjectHandles(void);





int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPWSTR    lpCmdLine,
    _In_ int       nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    // TODO: Place code here.

    // Initialize global strings
    wcscpy_s(szTitle, L"Histogram Project");
    wcscpy_s(szWindowClass, L"HistogramWindowClass");

    MyRegisterClass(hInstance);

    // Perform application initialization:
    if (!InitInstance(hInstance, nCmdShow))
    {
        return FALSE;
    }

    HACCEL hAccelTable = nullptr;

    MSG msg = { 0 };

    if(!resource.Initialize(hWnd))
    {
        fprintf(stderr, "core initialize failed\n");
        return FALSE;
    }

    SetObjectHandles();
    Update();

    // Main message loop:
    while (WM_QUIT != msg.message)
    {
        if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        Draw();
    }

    CloseObjectHandles();
    return (int)msg.wParam;
}



void SetObjectHandles(void)
{

    if (!imageProcess.Initialize(resource))
    {
        fprintf(stderr, "hitogram initialize failed\n");
#ifdef _DEBUG
        __debugbreak();
#endif
    }
    imageProcess.SetPSID(PS_HISTOGRAM);
    imageProcess.SetVSID(VS_HISTOGRAM);

}

void Update(void)
{
   
}

void Draw(void)
{
    imageProcess.Draw();

    resource.GetSwapChain()->Present(1, 0);
}

void CloseObjectHandles(void)
{
    imageProcess.CloseImgProcessHandles();
    
    resource.CloseD3DHandles();
}

















//  FUNCTION: MyRegisterClass()
//
//  PURPOSE: Registers the window class.
//
ATOM MyRegisterClass(HINSTANCE hInstance)
{
    WNDCLASSEXW wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = WndProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = hInstance;
    wcex.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
    wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszMenuName = nullptr;
    wcex.lpszClassName = szWindowClass;
    wcex.hIconSm = LoadIcon(nullptr, IDI_APPLICATION);

    return RegisterClassExW(&wcex);
}

//
//   FUNCTION: InitInstance(HINSTANCE, int)
//
//   PURPOSE: Saves instance handle and creates main window
//
//   COMMENTS:
//
//        In this function, we save the instance handle in a global variable and
//        create and display the main program window.
//
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
    hInst = hInstance; // Store instance handle in our global variable

    const int winW = 512;
    const int winH = 512;

    hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, winW, winH, nullptr, nullptr, hInstance, nullptr);

    if (!hWnd)
    {
        return FALSE;
    }

    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);

    return TRUE;
}

//
//  FUNCTION: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  PURPOSE: Processes messages for the main window.
//
//  WM_COMMAND  - process the application menu
//  WM_PAINT    - Paint the main window
//  WM_DESTROY  - post a quit message and return
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_COMMAND:
    {
        int wmId = LOWORD(wParam);
        // Parse the menu selections
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    break;
    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hWnd, &ps);
        // TODO: Add any drawing code that uses hdc here...
        EndPaint(hWnd, &ps);
    }
    break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    case WM_CHAR:
        switch (wParam)
        {
        case'b':
        case'B':
            imageProcess.SetVSID(VS_BLUR);
            imageProcess.SetPSID(PS_BLUR);
            break;

        case'h':
        case'H':
            imageProcess.SetPSID(PS_HISTOGRAM);
            imageProcess.SetVSID(VS_HISTOGRAM);
            break;
        case'e':
        case'E':
            imageProcess.SetVSID(VS_EQUALIZE);
            imageProcess.SetPSID(PS_EQUALIZE);
            break;
        case'c':
        case'C':
            imageProcess.SetVSID(VS_COLOR_MAP);
            imageProcess.SetPSID(PS_COLOR_MAP);
            break;
        case'l':
        case'L':
            imageProcess.SetVSID(VS_LAPLACIAN);
            imageProcess.SetPSID(PS_LAPLACIAN);
            break;
        case'f':
        case'F':
            imageProcess.SetVSID(VS_FUZZY_CONTRAST);
            imageProcess.SetPSID(PS_FUZZY_CONTRAST);
            break;
        case'm':
        case'M':
            imageProcess.SetVSID(VS_MULTI_FUZZY);
            imageProcess.SetPSID(PS_MULTI_FUZZY);
            break;
        case'd':
        case'D':
            imageProcess.SetVSID(VS_FOURIER_DFT);
            imageProcess.SetPSID(PS_FOURIER_DFT);
            break;
        case'o':
        case'O':
            imageProcess.SetVSID(VS_HOUGH);
            imageProcess.SetPSID(PS_HOUGH);
            break;
        case'w':
        case'W':
            imageProcess.SetVSID(VS_FOURIER_FFT);
            imageProcess.SetPSID(PS_FOURIER_FFT);
            break;
        }

    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

// Message handler for about box.
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(lParam);
    switch (message)
    {
    case WM_INITDIALOG:
        return (INT_PTR)TRUE;

    case WM_COMMAND:
        if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
        {
            EndDialog(hDlg, LOWORD(wParam));
            return (INT_PTR)TRUE;
        }
        break;
    }
    return (INT_PTR)FALSE;
}
