! program mpi_version_test
!     use mpi
!     integer :: ieer,ver,subver
!     print *,mpi_version, mpi_subversion
!     call mpi_get_version(ver,subver,ieer)
!     print *,ver,subver
!     end program mpi_version_test

! program mpidetest 
!     use mpi
!     integer :: ierr
!     call mpi_init(ierr)
!     print *,'i am using mpi'
!     call mpi_finalize(ierr)
! end program mpidetest


program main
    use mpi
    implicit none
    integer,parameter :: num=100, nccp=4
    integer,save :: myid, numprocs, namelen, rc, ierr
    integer,save :: istat(MPI_STATUS_SIZE)
    integer(8) :: i, objval, temp_objval
    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)
    objval=0
    temp_objval=0
    if(myid.ne.0) then
        do i=(myid-1)*(num/nccp)+1, myid*(num/nccp)
           temp_objval=temp_objval+i
        end do
        write(*,10) myid, temp_objval
10        format('Process',I2,' calculation is :',I6)
        call MPI_SEND(temp_objval, 1, MPI_INTEGER8, 0, mpi_tag,MPI_COMM_WORLD, ierr)
    else
        do i=1, nccp
           call MPI_RECV(temp_objval, 1, MPI_INTEGER8, i,mpi_any_tag, MPI_COMM_WORLD, istat, ierr)
           objval=temp_objval+objval
        end do
        write(*,20) objval
20        format('The final result is :',I6)
    end if
    call MPI_FINALIZE(rc)
end program main
